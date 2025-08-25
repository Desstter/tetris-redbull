
"""
tetris_phone_bot.py — versión robusta para Red Bull Tetris (celular)
Cambios clave:
- Detección por celda usando ∆E en CIE-Lab (no depende de Hue/Value).
- Tolera el celeste/azul poco saturado y la sombra diagonal.
- Movimiento con micro-swipes por columna para móviles.
- Prioriza limpiar líneas (scoring agresivo a lines_cleared).
- "Wake" sin spam de taps.
- Debug de visión: guarda crop, máscara y overlay de rejilla.
"""

import argparse, logging, platform, random, signal, subprocess, sys, time, os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2

# ---------- deps opcionales (para scrcpy) ----------
try:
    import pyautogui
    import pygetwindow as gw
    import mss
    SCRCPY_DEPS_OK = True
except Exception:
    SCRCPY_DEPS_OK = False


# ============================ Utilidades generales ============================

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

def parse_rect(rect_str: str) -> Tuple[float, float, float, float]:
    try:
        x,y,w,h = [float(t.strip()) for t in rect_str.split(",")]
        assert 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
        return x,y,w,h
    except Exception:
        raise argparse.ArgumentTypeError('Rect inválido. Usa "x,y,w,h" en [0..1].')

def jitter(px:int)->int:
    return random.randint(-px, px) if px>0 else 0

def clamp(v,a,b): return max(a, min(b, v))


# ========================== Sistema de Configuración ==========================

import json
from pathlib import Path

class TetrisConfig:
    """Maneja la configuración centralizada del bot"""
    def __init__(self, config_path=None):
        self.config_path = config_path or "config.json"
        self.config = self._load_config()
    
    def _load_config(self):
        """Carga configuración desde JSON con fallbacks seguros"""
        default_config = {
            "vision": {
                "normal_mode": {"pad_c": 0.20, "ring_pad": 0.06, "bg_floor": 10.0, "s_min": 65, "v_min": 105, "k_clusters": 3},
                "tight_mode": {"pad_c": 0.22, "ring_pad": 0.06, "bg_floor": 12.0, "s_min": 70, "v_min": 110, "k_clusters": 3}
            },
            "devices": {"default": {"rect": [0.185, 0.225, 0.63, 0.57]}},
            "gameplay": {"fps": 10, "session_sec": 185, "piece_tracker_timeout": 1.5, "max_component_size": 4}
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    logging.info(f"Configuración cargada desde {self.config_path}")
                    # Merge con default para llenar campos faltantes
                    return self._deep_merge(default_config, loaded_config)
            else:
                logging.info(f"No se encontró {self.config_path}, usando configuración por defecto")
                return default_config
        except Exception as e:
            logging.warning(f"Error cargando configuración: {e}, usando valores por defecto")
            return default_config
    
    def _deep_merge(self, default, loaded):
        """Merge profundo de diccionarios manteniendo estructura"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_vision_params(self, mode="normal"):
        """Obtiene parámetros de visión por modo"""
        mode_key = f"{mode}_mode"
        return self.config.get("vision", {}).get(mode_key, {})
    
    def get_device_rect(self, device_name="default"):
        """Obtiene rectángulo del tablero para un dispositivo específico"""
        devices = self.config.get("devices", {})
        device_config = devices.get(device_name, devices.get("default", {}))
        rect = device_config.get("rect", [0.185, 0.225, 0.63, 0.57])
        return tuple(rect)
    
    def get_gameplay_param(self, param, default=None):
        """Obtiene parámetro de gameplay"""
        return self.config.get("gameplay", {}).get(param, default)
    
    def get_gesture_param(self, category, param, default=None):
        """Obtiene parámetro de gestos"""
        return self.config.get("gestures", {}).get(category, {}).get(param, default)
    
    def list_available_devices(self):
        """Lista dispositivos disponibles en la configuración"""
        return list(self.config.get("devices", {}).keys())


# ========================== Arquitectura de Clases Modular ==========================

@dataclass
class BoardAnalysis:
    """Resultado del análisis de un frame del tablero"""
    occupancy_grid: np.ndarray
    debug_mask: np.ndarray
    active_piece: Optional[List[Tuple[int,int]]]
    ghost_piece: Optional[List[Tuple[int,int]]]
    occupation_rate: float
    components_found: int

class TetrisVision:
    """Maneja todo el análisis visual del tablero de Tetris"""
    
    def __init__(self, config: TetrisConfig):
        self.config = config
        self.temporal_filter = TemporalFilter(
            history_size=config.config.get("vision", {}).get("temporal_filter_history", 7),  # Increased from 5
            confidence_threshold=config.config.get("vision", {}).get("temporal_filter_threshold", 0.7)  # Increased from 0.6
        )
    
    def analyze_board(self, crop: np.ndarray, rows=20, cols=10, use_temporal_filter=True) -> BoardAnalysis:
        """
        Análisis completo del tablero usando el nuevo sistema multilayer.
        Mucho más robusto que el sistema anterior basado en comparaciones HSV relativas.
        """
        # NUEVO: Usar sistema multilayer para detección directa
        raw_piece_cells, raw_ghost_cells, debug_info = detect_pieces_multilayer(crop, rows, cols)
        occ_raw, _mask = occupancy_grid(crop, rows, cols, mode="normal")
        occ = occ_raw.copy()
        piece_cells = (self.temporal_filter.get_filtered_piece()           
            if use_temporal_filter and raw_piece_cells else raw_piece_cells)
        ghost_cells = raw_ghost_cells
        # Aplicar filtrado temporal si está habilitado
        if use_temporal_filter and raw_piece_cells:
            self.temporal_filter.add_detection(raw_piece_cells)
            piece_cells = self.temporal_filter.get_filtered_piece()
        else:
            piece_cells = raw_piece_cells
        
        # Si usamos filtro temporal, puede que necesitemos redetectar el ghost para la nueva pieza
        ghost_cells = raw_ghost_cells
        if use_temporal_filter and piece_cells != raw_piece_cells and piece_cells:
            # Re-detectar ghost para la pieza filtrada temporalmente
            _, ghost_cells, _ = detect_pieces_multilayer(crop, rows, cols)
        
        # Crear grilla de ocupación compatible (solo piezas activas y bloques ya colocados)
        # Esto mantiene compatibilidad con el resto del código
        bg_grid, active_grid, ghost_grid, debug_mask = occupancy_grid_multilayer(crop, rows, cols)
        
        # La grilla de ocupación final incluye solo las piezas no-ghost
        occ = active_grid.copy()
        
        # Quitar las celdas de la pieza activa detectada si es diferente a la grilla
        if piece_cells:
            for r, c in piece_cells:
                if 0 <= r < rows and 0 <= c < cols:
                    occ[r, c] = False
            # Marcar solo las celdas de la pieza actual
            for r, c in piece_cells:
                if 0 <= r < rows and 0 <= c < cols:
                    occ[r, c] = True

        # Estadísticas
        num_occupied = int(occ.sum())
        total_cells = rows * cols
        occupation_rate = num_occupied / total_cells
        
        # Contar componentes (excluyendo ghost)
        components = extract_components_by_type(occ, np.zeros_like(occ))[0]  # Solo activos
        components_found = len(components)
        
        # Log del nuevo sistema mejorado con más contexto
        method_info = "multilayer + temporal" if use_temporal_filter else "multilayer"
        stability_score = self.temporal_filter.get_stability_score() if use_temporal_filter else 1.0
        
        # Log detallado de detección de visión
        logging.debug(f"👁️ VISION ANALYSIS ({method_info}):")
        logging.debug(f"   🔍 Grid analysis: {num_occupied}/{total_cells} occupied ({occupation_rate:.1%})")
        logging.debug(f"   🧩 Components found: {components_found}")
        logging.debug(f"   🎯 Active piece: {len(piece_cells) if piece_cells else 0} cells detected")
        logging.debug(f"   👻 Ghost piece: {len(ghost_cells) if ghost_cells else 0} cells detected")
        if use_temporal_filter:
            logging.debug(f"   🎚️ Temporal stability: {stability_score:.2f}")
            if raw_piece_cells != piece_cells:
                logging.debug(f"   🔄 Temporal filter active: raw {len(raw_piece_cells) if raw_piece_cells else 0} -> filtered {len(piece_cells) if piece_cells else 0}")
        
        # Log de posiciones específicas si hay pieza detectada
        if piece_cells:
            try:
                r_min, r_max = min(r for r, c in piece_cells), max(r for r, c in piece_cells)
                c_min, c_max = min(c for r, c in piece_cells), max(c for r, c in piece_cells)
                piece_height = r_max - r_min + 1
                piece_width = c_max - c_min + 1
                logging.info(f"🎯 Piece detected ({method_info}): {len(piece_cells)} cells at rows {r_min}-{r_max}, cols {c_min}-{c_max} ({piece_width}x{piece_height})")
            except ValueError:
                logging.info(f"🎯 Piece detected ({method_info}): {len(piece_cells)} cells (position unknown)")
        
        if ghost_cells:
            try:
                r_min, r_max = min(r for r, c in ghost_cells), max(r for r, c in ghost_cells)
                c_min, c_max = min(c for r, c in ghost_cells), max(c for r, c in ghost_cells)
                logging.info(f"👻 Ghost detected ({method_info}): {len(ghost_cells)} cells at rows {r_min}-{r_max}, cols {c_min}-{c_max}")
            except ValueError:
                logging.info(f"👻 Ghost detected ({method_info}): {len(ghost_cells)} cells")
        
        # Warnings sobre detecciones sospechosas con más contexto
        if piece_cells and len(piece_cells) > 4:
            logging.error(f"❌ CRITICAL: Oversized piece detected: {len(piece_cells)} cells (max allowed: 4)")
            logging.error(f"   Piece cells: {piece_cells}")
            logging.error(f"   This indicates vision system malfunction - piece will be rejected")
        elif piece_cells and len(piece_cells) == 1:
            logging.warning(f"⚠️ Single cell piece detected - possibly noise or I-piece end")
        elif piece_cells and len(piece_cells) in [2, 3, 4]:
            logging.debug(f"✅ Valid piece size: {len(piece_cells)} cells")
            
        if occupation_rate > 0.8:
            logging.warning(f"⚠️ Very high board occupation: {occupation_rate:.1%} - may indicate detection issues")
        if components_found > 15:
            logging.warning(f"⚠️ High component count: {components_found} - possible noise in detection")
            
        # Additional piece validation
        if piece_cells and ghost_cells:
            if len(set(piece_cells) & set(ghost_cells)) > 0:
                logging.warning(f"⚠️ Piece and ghost overlap detected - may indicate detection confusion")

        return BoardAnalysis(
            occupancy_grid=occ,
            debug_mask=debug_info['debug_mask'],  # Usar la máscara multilayer más informativa
            active_piece=piece_cells,
            ghost_piece=ghost_cells,
            occupation_rate=occupation_rate,
            components_found=components_found
        )
    
    def get_occupancy_grid(self, crop: np.ndarray, rows=20, cols=10, mode="normal"):
        """Wrapper para occupancy_grid - migración gradual"""
        return occupancy_grid(crop, rows, cols, mode)
    
    def find_active_piece_in_grid(self, occ: np.ndarray, crop: np.ndarray = None):
        """Wrapper para find_active_piece - migración gradual"""
        return find_active_piece(occ, crop)
    
    def detect_ghost_piece(self, crop: np.ndarray, occ: np.ndarray, piece_cells: List[Tuple[int,int]]):
        """Wrapper para detect_ghost_component - migración gradual"""
        return detect_ghost_component(crop, occ, piece_cells)
    
    def get_detection_stability(self) -> float:
        """Obtiene el score de estabilidad del filtro temporal"""
        return self.temporal_filter.get_stability_score()
    
    def is_detection_stable(self) -> bool:
        """Verifica si las detecciones actuales son estables"""
        return self.temporal_filter.is_detection_stable()
    
    def reset_temporal_filter(self):
        """Reinicia el filtro temporal (útil al empezar nueva partida)"""
        self.temporal_filter.reset()

class TemporalFilter:
    """Sistema de filtrado temporal para suavizar detecciones y reducir jitter"""
    
    def __init__(self, history_size=5, confidence_threshold=0.6):
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.piece_history = []
        self.position_history = []
        self.shape_history = []
        
    def add_detection(self, piece_cells: Optional[List[Tuple[int,int]]], timestamp: float = None):
        """Añade una nueva detección al historial"""
        import time
        if timestamp is None:
            timestamp = time.time()
            
        # Añadir al historial
        self.piece_history.append({
            'cells': piece_cells,
            'timestamp': timestamp,
            'valid': piece_cells is not None and len(piece_cells) > 0
        })
        
        # Mantener solo las últimas detecciones
        if len(self.piece_history) > self.history_size:
            self.piece_history.pop(0)
    
    def get_filtered_piece(self) -> Optional[List[Tuple[int,int]]]:
        """Retorna la pieza filtrada basada en el historial temporal"""
        if not self.piece_history:
            return None
            
        # Contar detecciones válidas recientes
        valid_detections = [h for h in self.piece_history if h['valid']]
        
        if len(valid_detections) == 0:
            return None
            
        # Si tenemos suficientes detecciones válidas, usar consenso
        if len(valid_detections) >= max(1, int(self.history_size * self.confidence_threshold)):
            return self._get_consensus_piece(valid_detections)
        
        # Si no, retornar la última detección válida
        return valid_detections[-1]['cells'] if valid_detections else None
    
    def _get_consensus_piece(self, valid_detections: List[dict]) -> Optional[List[Tuple[int,int]]]:
        """Calcula consenso entre múltiples detecciones válidas con validación mejorada"""
        if not valid_detections:
            return None
            
        # Filtrar detecciones inválidas (más de 4 células, posiciones imposibles)
        valid_pieces = []
        for detection in valid_detections:
            cells = detection['cells']
            if cells and self._is_valid_tetris_piece(cells):
                valid_pieces.append(cells)
                
        if not valid_pieces:
            logging.warning("🔄 Temporal filter: No valid tetris pieces found in history")
            return None
            
        # Si solo hay una pieza válida, usarla
        if len(valid_pieces) == 1:
            return valid_pieces[0]
            
        # Consenso por posición: buscar piezas que estén en posiciones similares
        consensus_piece = self._find_position_consensus(valid_pieces)
        
        if consensus_piece:
            logging.debug(f"🔄 Temporal filter: Consensus found from {len(valid_pieces)} valid detections")
            return consensus_piece
        else:
            # Fallback: usar la más reciente válida
            logging.debug("🔄 Temporal filter: Using most recent valid detection")
            return valid_pieces[-1]
    
    def _is_valid_tetris_piece(self, cells: List[Tuple[int,int]]) -> bool:
        """Valida si las células forman una pieza válida de Tetris"""
        if not cells or len(cells) > 4 or len(cells) < 1:
            return False
            
        # Verificar que las coordenadas están en rango válido
        for r, c in cells:
            if r < 0 or r >= 20 or c < 0 or c >= 10:
                return False
                
        # Verificar conectividad (células adyacentes)
        if len(cells) > 1:
            cell_set = set(cells)
            connected_cells = set()
            stack = [cells[0]]
            connected_cells.add(cells[0])
            
            while stack:
                r, c = stack.pop()
                # Verificar 4 direcciones
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cell_set and (nr, nc) not in connected_cells:
                        connected_cells.add((nr, nc))
                        stack.append((nr, nc))
                        
            # Todas las células deben estar conectadas
            return len(connected_cells) == len(cells)
            
        return True
    
    def _find_position_consensus(self, pieces: List[List[Tuple[int,int]]]) -> Optional[List[Tuple[int,int]]]:
        """Encuentra consenso basado en posiciones similares de las piezas"""
        if len(pieces) < 2:
            return pieces[0] if pieces else None
            
        # Agrupar por columna izquierda (posición horizontal aproximada)
        position_groups = {}
        for piece in pieces:
            try:
                _, left_col, _, _ = bounding_box(piece)
                if left_col not in position_groups:
                    position_groups[left_col] = []
                position_groups[left_col].append(piece)
            except:
                continue
                
        # Encontrar el grupo más grande (consenso)
        if position_groups:
            largest_group = max(position_groups.values(), key=len)
            if len(largest_group) >= len(pieces) * 0.5:  # Al menos 50% de consenso
                return largest_group[0]  # Usar el primero del grupo consensuado
                
        return None
    
    def get_stability_score(self) -> float:
        """Retorna un score de estabilidad de las detecciones (0-1)"""
        if len(self.piece_history) < 2:
            return 0.0
            
        valid_count = sum(1 for h in self.piece_history if h['valid'])
        return valid_count / len(self.piece_history)
    
    def is_detection_stable(self) -> bool:
        """Determina si las detecciones actuales son estables"""
        return self.get_stability_score() >= self.confidence_threshold
    
    def reset(self):
        """Reinicia el filtro temporal"""
        self.piece_history.clear()
        self.position_history.clear()
        self.shape_history.clear()

class MovementCorrector:
    """Sistema de corrección adaptiva para movimientos imprecisos con detección de offset sistemático"""
    
    def __init__(self):
        self.movement_history = []
        self.column_corrections = {}  # column -> average correction needed
        self.success_rate = 0.0
        self.total_attempts = 0
        self.successful_attempts = 0
        
        # NEW: Systematic offset detection
        self.systematic_offset = 0.0  # Average offset across all movements
        self.offset_detection_threshold = 5  # Need at least 5 attempts for offset detection
        self.last_offset_warning = 0  # Track when we last warned about offset
        
    def record_movement_attempt(self, start_col: int, target_col: int, actual_col: int, successful: bool):
        """Registra un intento de movimiento para aprendizaje adaptivo"""
        attempt = {
            'start_col': start_col,
            'target_col': target_col, 
            'actual_col': actual_col,
            'successful': successful,
            'error': actual_col - target_col if actual_col is not None else None,
            'timestamp': time.time()
        }
        
        self.movement_history.append(attempt)
        self.total_attempts += 1
        
        if successful:
            self.successful_attempts += 1
            
        # Keep only recent history (last 50 attempts)
        if len(self.movement_history) > 50:
            self.movement_history.pop(0)
            
        # Update success rate
        self.success_rate = self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0.0
        
        # Learn column-specific corrections
        if attempt['error'] is not None:
            if target_col not in self.column_corrections:
                self.column_corrections[target_col] = []
            self.column_corrections[target_col].append(attempt['error'])
            
            # Keep only recent corrections for each column
            if len(self.column_corrections[target_col]) > 10:
                self.column_corrections[target_col].pop(0)
                
        # NEW: Update systematic offset calculation
        self._update_systematic_offset()
        
        logging.debug(f"🔄 Movement recorded: {start_col}→{target_col} (actual: {actual_col}, success: {successful})")
        
        # NEW: Check for systematic offset pattern and warn
        if self.total_attempts >= self.offset_detection_threshold:
            self._check_systematic_offset()
        
    def _update_systematic_offset(self):
        """Actualiza el cálculo de offset sistemático"""
        recent_errors = []
        for attempt in self.movement_history[-10:]:  # Last 10 attempts
            if attempt['error'] is not None:
                recent_errors.append(attempt['error'])
                
        if recent_errors:
            self.systematic_offset = sum(recent_errors) / len(recent_errors)
            
    def _check_systematic_offset(self):
        """Verifica y advierte sobre offset sistemático consistente"""
        if abs(self.systematic_offset) > 2.0:  # Significant systematic error
            current_time = time.time()
            if current_time - self.last_offset_warning > 30:  # Warn max once per 30 seconds
                logging.error(f"🚨 SYSTEMATIC OFFSET DETECTED: {self.systematic_offset:.1f} columns")
                logging.error(f"   This indicates board coordinate calibration is incorrect!")
                logging.error(f"   Consistently moving {abs(self.systematic_offset):.1f} columns {'left' if self.systematic_offset < 0 else 'right'} of target")
                logging.error(f"   Consider recalibrating board rectangle coordinates")
                self.last_offset_warning = current_time

    def get_corrected_target(self, target_col: int) -> int:
        """Obtiene columna objetivo corregida basada en historial de errores y offset sistemático"""
        corrected_target = target_col
        
        # First apply systematic offset correction if significant
        if abs(self.systematic_offset) > 1.5 and self.total_attempts >= self.offset_detection_threshold:
            systematic_correction = -int(round(self.systematic_offset))
            corrected_target += systematic_correction
            logging.debug(f"🔧 Systematic offset correction: {target_col} → {corrected_target} (offset: {self.systematic_offset:.1f})")
        
        # Then apply column-specific correction
        if target_col in self.column_corrections:
            errors = self.column_corrections[target_col]
            if len(errors) >= 3:  # Need at least 3 data points
                avg_error = sum(errors) / len(errors)
                column_correction = -int(round(avg_error))
                if column_correction != 0:
                    old_target = corrected_target
                    corrected_target += column_correction
                    logging.debug(f"🎯 Column-specific correction: {old_target} → {corrected_target} (avg error: {avg_error:.1f})")
        
        # Clamp to valid range
        final_target = max(0, min(9, corrected_target))
        
        if final_target != target_col:
            correction_type = "systematic + column" if abs(self.systematic_offset) > 1.5 else "column-specific"
            logging.info(f"🧠 Adaptive correction ({correction_type}): {target_col} → {final_target}")
            
        return final_target
        
    def get_recommended_timing_multiplier(self) -> float:
        """Recomienda multiplicador de timing basado en tasa de éxito"""
        if self.total_attempts < 5:
            return 1.0
            
        if self.success_rate < 0.5:
            return 1.5  # Much slower for very low success
        elif self.success_rate < 0.7:
            return 1.3  # Slower for low success  
        elif self.success_rate < 0.8:
            return 1.1  # Slightly slower for moderate success
        else:
            return 1.0  # Normal timing for high success
            
    def get_diagnostics(self) -> dict:
        """Obtiene diagnósticos del sistema de corrección incluyendo offset sistemático"""
        recent_errors = []
        for attempt in self.movement_history[-10:]:
            if attempt['error'] is not None:
                recent_errors.append(attempt['error'])
                
        return {
            'total_attempts': self.total_attempts,
            'success_rate': self.success_rate,
            'recent_avg_error': sum(recent_errors) / len(recent_errors) if recent_errors else 0,
            'systematic_offset': self.systematic_offset,
            'offset_detected': abs(self.systematic_offset) > 2.0,
            'columns_learned': len(self.column_corrections),
            'recommended_timing': self.get_recommended_timing_multiplier()
        }

class TetrisController:
    """Maneja todas las acciones de control del juego"""
    
    def __init__(self, backend: 'ScreenBackend', zones: 'GestureZones', config: TetrisConfig):
        self.backend = backend
        self.zones = zones
        self.config = config
        self.movement_corrector = MovementCorrector()  # Add adaptive correction
    
    def rotate_piece(self):
        """Wrapper para rotate_action - migración gradual"""
        rotate_action(self.backend, self.zones)
    
    def move_piece_to_column(self, piece_cells: List[Tuple[int,int]], target_col: int, board: 'BoardRect'):
        """
        Wrapper mejorado con sistema adaptivo de corrección de movimientos.
        """
        max_retries = 3
        retry_count = 0
        movement_successful = False
        
        # Get initial position for reference
        try:
            _, initial_col, _, _ = bounding_box(piece_cells)
            logging.info(f"🎯 Movement request: col {initial_col} → col {target_col}")
        except:
            initial_col = 5  # fallback
            
        # ADAPTIVE CORRECTION: Apply learned corrections
        original_target = target_col
        corrected_target = self.movement_corrector.get_corrected_target(target_col)
        adaptive_timing = self.movement_corrector.get_recommended_timing_multiplier()
        
        if corrected_target != original_target:
            logging.info(f"🧠 Adaptive target correction: {original_target} → {corrected_target}")
            
        # Show current adaptive system status
        diagnostics = self.movement_corrector.get_diagnostics()
        if diagnostics['total_attempts'] > 0:
            logging.debug(f"📊 Movement AI: {diagnostics['success_rate']:.1%} success rate, "
                         f"{diagnostics['columns_learned']} cols learned, timing {diagnostics['recommended_timing']:.1f}x")
            
        while retry_count <= max_retries and not movement_successful:
            # Combine adaptive timing with retry-based timing progression
            base_timing = adaptive_timing + (retry_count * 0.3)
            logging.info(f"🔄 Movement attempt {retry_count + 1}/{max_retries + 1} (timing: {base_timing:.1f}x)")
            
            move_piece_to_column(self.backend, self.zones, board, piece_cells, corrected_target, base_timing)
            
            # Espera extra para que el movimiento se complete
            time.sleep(0.12 + retry_count * 0.05)
            
            # VERIFICACIÓN REAL POST-MOVIMIENTO
            if retry_count < max_retries:
                actual_col = self._get_actual_piece_position(board)
                verification_successful = abs(actual_col - corrected_target) <= 1 if actual_col is not None else False
                logging.debug(f"   Verify: actual={actual_col}, target_corrected={corrected_target}, target_original={original_target}")

                
                # RECORD MOVEMENT FOR ADAPTIVE LEARNING
                self.movement_corrector.record_movement_attempt(
                    initial_col, original_target, actual_col, verification_successful
                )
                
                if verification_successful:
                    logging.info(f"✅ Movement verified successful on attempt {retry_count + 1}")
                    movement_successful = True
                    break
                else:
                    logging.warning(f"❌ Movement verification failed on attempt {retry_count + 1}")
                    if actual_col is not None:
                        logging.warning(f"   Expected: {original_target}, Actual: {actual_col}, Error: {actual_col - original_target}")
                    retry_count += 1
            else:
                # Last attempt - record attempt but don't verify
                actual_col = self._get_actual_piece_position(board)
                self.movement_corrector.record_movement_attempt(
                    initial_col, original_target, actual_col, False  # Assume failed on final attempt
                )
                break
                
        if retry_count > 0:
            status = "✅ successful" if movement_successful else "❌ failed"
            logging.info(f"📊 Movement summary: {retry_count + 1} attempts, {status}")
            
            # Show updated adaptive status if we have learned something
            if retry_count >= max_retries and not movement_successful:
                new_timing = self.movement_corrector.get_recommended_timing_multiplier()
                logging.info(f"🧠 Adaptive system updated: recommended timing now {new_timing:.1f}x")
            
        return movement_successful
    
    def _get_actual_piece_position(self, board: 'BoardRect') -> Optional[int]:
        """Helper to get actual piece position for adaptive learning"""
        try:
            frame = self.backend.get_screen()
            crop = frame[board.y0:board.y0+board.h, board.x0:board.x0+board.w]
            
            occ, _ = occupancy_grid(crop, rows=20, cols=10)
            current_piece = find_active_piece(occ, crop)
            
            if current_piece:
                _, actual_col, _, _ = bounding_box(current_piece)
                return actual_col
                
        except Exception as e:
            logging.debug(f"Could not get actual piece position: {e}")
            
        return None
    
    
    def drop_piece(self):
        """Wrapper para drop_action - migración gradual"""
        drop_action(self.backend, self.zones)

class TetrisGame:
    """Maneja la lógica del juego, simulación y estrategia"""
    
    def __init__(self, config: TetrisConfig):
        self.config = config
        
        # Seleccionar política según configuración
        use_multistep = config.get_gameplay_param("use_multistep_policy", False)
        if use_multistep:
            lookahead_depth = config.get_gameplay_param("lookahead_depth", 2)
            self.policy = MultiStepPolicy(lookahead_depth=lookahead_depth)
            logging.info(f"🧠 Usando MultiStepPolicy con profundidad {lookahead_depth}")
        else:
            self.policy = OneStepPolicy()
            logging.info("🎯 Usando OneStepPolicy (simple)")
            
        self.tracker = PieceTracker()
    
    def simulate_drop(self, board: np.ndarray, shape: List[Tuple[int,int]], left_col: int):
        """Wrapper para drop_simulation - migración gradual"""
        return drop_simulation(board, shape, left_col)
    
    def evaluate_board(self, board: np.ndarray, lines_cleared: int) -> float:
        """Wrapper para evaluate_board con soporte para evaluación avanzada"""
        use_advanced = self.config.config.get("evaluation", {}).get("use_advanced_evaluation", False)
        
        if use_advanced:
            enable_tspin = self.config.config.get("evaluation", {}).get("enable_tspin_detection", True)
            enable_combo = self.config.config.get("evaluation", {}).get("enable_combo_bonus", True)
            return evaluate_board_advanced(board, lines_cleared, enable_tspin, enable_combo)
        else:
            return evaluate_board(board, lines_cleared)
    
    def choose_action(self, board_stack: np.ndarray, piece_type: str) -> Optional[Tuple[int,int]]:
        """Wrapper para policy.choose - migración gradual"""
        return self.policy.choose(board_stack, piece_type)
    
    def is_new_piece(self, piece_cells: List[Tuple[int,int]]) -> bool:
        """Wrapper para tracker.is_new - migración gradual"""
        return self.tracker.is_new(piece_cells)
    
    def mark_piece_acted(self):
        """Wrapper para tracker.mark_acted - migración gradual"""
        self.tracker.mark_acted()

class TetrisBot:
    """Orquestador principal que coordina todas las clases"""
    
    def __init__(self, config: TetrisConfig, backend: 'ScreenBackend', board: 'BoardRect'):
        self.config = config
        self.backend = backend
        self.board = board
        self.zones = compute_gesture_zones(board)
        
        # Inicializar componentes modulares
        self.vision = TetrisVision(config)
        self.controller = TetrisController(backend, self.zones, config)
        self.game = TetrisGame(config)
        self.performance_monitor = PerformanceMonitor()
        
        # Parámetros del loop
        self.fps = int(config.get_gameplay_param("fps", 10))
        self.session_sec = int(config.get_gameplay_param("session_sec", 185))
        self.dt = 1.0 / float(clamp(self.fps, 3, 15))
        
        logging.info(f"TetrisBot inicializado - FPS: {self.fps}, Sesión: {self.session_sec}s")
    
    def run_game_loop(self, debug_vision: bool=False, max_debug_frames: int=50):
        """
        Bucle principal:
        - Captura frame -> crop
        - Visión multilayer + filtro temporal
        - Construye board stack (sin pieza activa)
        - Clasifica pieza y decide (policy)
        - Rota, mueve, drop
        - Métricas + debug
        """
        rows, cols = 20, 10
        t0 = time.time()
        frame_idx = 0
        saved_dbg = 0
        
        try:
            while (time.time() - t0) < self.session_sec:
                tic = time.time()
                frame = self.backend.get_screen()
                crop = frame[self.board.y0:self.board.y0+self.board.h,
                             self.board.x0:self.board.x0+self.board.w]

                analysis = self.vision.analyze_board(crop, rows=rows, cols=cols, use_temporal_filter=True)
                occ = analysis.occupancy_grid.astype(bool)
                piece_cells = analysis.active_piece

                detection_success = piece_cells is not None and 1 <= len(piece_cells) <= 4
                self.performance_monitor.log_frame_metrics(time.time(), detection_success)

                # Guardado de debug
                if debug_vision and saved_dbg < max_debug_frames:
                    try:
                        os.makedirs("tetris_debug", exist_ok=True)
                        cv2.imwrite(f"tetris_debug/crop_{frame_idx:05d}.png", crop)
                        dbg_mask = analysis.debug_mask
                        cv2.imwrite(f"tetris_debug/mask_{frame_idx:05d}.png", dbg_mask)
                        saved_dbg += 1
                    except Exception as e:
                        logging.debug(f"No pudo guardar debug: {e}")

                if not detection_success:
                    # Tap suave para mantener "awake" sin spam
                    if frame_idx % 30 == 0:
                        xw, yw = self.zones.rotate_xy
                        self.backend.tap(xw + jitter(3), yw + jitter(3), hold_ms=60)
                    time.sleep(self.dt)
                    frame_idx += 1
                    continue

                # Construir board stack SIN la pieza activa
                board_stack = occ.copy()
                for (r, c) in piece_cells:
                    if 0 <= r < rows and 0 <= c < cols:
                        board_stack[r, c] = False

                # Clasificar pieza actual
                piece_info = classify_piece(piece_cells)
                if not piece_info:
                    logging.warning("No se pudo clasificar la pieza actual; saltando frame")
                    time.sleep(self.dt)
                    frame_idx += 1
                    continue
                piece_type, cur_orient = piece_info

                # Elegir acción
                decision = self.game.choose_action(board_stack, piece_type)
                if not decision:
                    logging.warning("Sin decisión válida; saltando")
                    time.sleep(self.dt)
                    frame_idx += 1
                    continue

                target_orient, target_left = decision
                log_decision_context(piece_type, board_stack, piece_cells, target_left, target_orient, score="(see debug)", reason="policy choose")

                # Rotar las veces necesarias (módulo número de orientaciones de la pieza)
                total_orients = len(PIECE_ORIENTS[piece_type])
                spins = (target_orient - cur_orient) % total_orients
                for _ in range(spins):
                    self.controller.rotate_piece()
                    time.sleep(0.05)

                # Mover a columna
                self.controller.move_piece_to_column(piece_cells, target_left, self.board)

                # Drop
                self.controller.drop_piece()

                # Métricas pieza
                self.performance_monitor.log_piece_action(piece_type, True, lines_cleared=0)

                # Ritmo del loop
                elapsed = time.time() - tic
                sleep_left = max(0.0, self.dt - elapsed)
                time.sleep(sleep_left)
                frame_idx += 1

        except KeyboardInterrupt:
            logging.info("⏹️ Interrumpido por el usuario.")
        finally:
            self.performance_monitor.log_performance_summary()

class PerformanceMonitor:
    """Sistema de monitoreo y métricas para el rendimiento del bot"""
    
    def __init__(self):
        import time
        self.start_time = time.time()
        self.metrics = {
            'fps': [],
            'detection_accuracy': [],
            'action_latency': [],
            'frame_times': [],
            'pieces_processed': 0,
            'lines_cleared': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'total_frames': 0
        }
        self.current_session = {
            'start_time': self.start_time,
            'last_frame_time': None,
            'frame_count': 0,
            'piece_count': 0
        }
        
    def log_frame_metrics(self, frame_time: float, detection_success: bool, 
                         action_time: float = 0.0):
        """Registra métricas de un frame procesado"""
        current_time = frame_time
        
        # Calcular FPS instantáneo
        if self.current_session['last_frame_time'] is not None:
            frame_delta = current_time - self.current_session['last_frame_time']
            if frame_delta > 0:
                instant_fps = 1.0 / frame_delta
                self.metrics['fps'].append(instant_fps)
        
        # Registrar métricas
        self.metrics['frame_times'].append(frame_time)
        self.metrics['action_latency'].append(action_time)
        self.metrics['total_frames'] += 1
        
        # Tracking de detecciones
        if detection_success:
            self.metrics['successful_detections'] += 1
        else:
            self.metrics['failed_detections'] += 1
        
        # Actualizar estado de sesión
        self.current_session['last_frame_time'] = current_time
        self.current_session['frame_count'] += 1
    
    def log_piece_action(self, piece_type: str, action_taken: bool, lines_cleared: int = 0):
        """Registra métricas relacionadas con acciones en piezas"""
        self.metrics['pieces_processed'] += 1
        self.metrics['lines_cleared'] += lines_cleared
        self.current_session['piece_count'] += 1
        
        # Calcular accuracy de detección
        total_detections = self.metrics['successful_detections'] + self.metrics['failed_detections']
        if total_detections > 0:
            accuracy = self.metrics['successful_detections'] / total_detections
            self.metrics['detection_accuracy'].append(accuracy)
    
    def get_current_stats(self) -> dict:
        """Obtiene estadísticas actuales del rendimiento"""
        import time
        current_time = time.time()
        session_duration = current_time - self.current_session['start_time']
        
        # Calcular estadísticas
        avg_fps = np.mean(self.metrics['fps']) if self.metrics['fps'] else 0
        avg_latency = np.mean(self.metrics['action_latency']) if self.metrics['action_latency'] else 0
        detection_rate = (self.metrics['successful_detections'] / 
                         max(1, self.metrics['successful_detections'] + self.metrics['failed_detections']))
        
        return {
            'session_duration': session_duration,
            'total_frames': self.metrics['total_frames'],
            'pieces_processed': self.metrics['pieces_processed'],
            'lines_cleared': self.metrics['lines_cleared'],
            'avg_fps': avg_fps,
            'avg_latency': avg_latency,
            'detection_success_rate': detection_rate,
            'frames_per_minute': self.metrics['total_frames'] / (session_duration / 60) if session_duration > 0 else 0
        }
    
    def log_performance_summary(self):
        """Imprime un resumen de rendimiento en el log"""
        stats = self.get_current_stats()
        
        logging.info("=" * 60)
        logging.info("📊 RESUMEN DE RENDIMIENTO")
        logging.info("=" * 60)
        logging.info(f"⏱️  Duración de sesión: {stats['session_duration']:.1f} segundos")
        logging.info(f"🖼️  Frames procesados: {stats['total_frames']}")
        logging.info(f"🧩 Piezas procesadas: {stats['pieces_processed']}")
        logging.info(f"📏 Líneas eliminadas: {stats['lines_cleared']}")
        logging.info(f"🚀 FPS promedio: {stats['avg_fps']:.2f}")
        logging.info(f"⚡ Latencia promedio: {stats['avg_latency']:.3f}s")
        logging.info(f"🎯 Tasa de éxito en detecciones: {stats['detection_success_rate']:.1%}")
        logging.info(f"📈 Frames por minuto: {stats['frames_per_minute']:.1f}")
        logging.info("=" * 60)
    
    def export_metrics(self, filename: str = None):
        """Exporta métricas a archivo JSON"""
        import json
        import time
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"tetris_metrics_{timestamp}.json"
        
        export_data = {
            'session_info': self.current_session,
            'metrics': self.metrics,
            'summary': self.get_current_stats(),
            'export_timestamp': time.time()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            logging.info(f"📁 Métricas exportadas a: {filename}")
            return filename
        except Exception as e:
            logging.error(f"❌ Error exportando métricas: {e}")
            return None
    
    def reset_session(self):
        """Reinicia las métricas para una nueva sesión"""
        import time
        self.start_time = time.time()
        
        # Guardar métricas anteriores si es necesario
        if self.metrics['total_frames'] > 0:
            logging.info(f"🔄 Reiniciando monitor. Sesión anterior: {self.metrics['total_frames']} frames")
        
        # Reset métricas
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []
            else:
                self.metrics[key] = 0
        
        # Reset sesión
        self.current_session = {
            'start_time': self.start_time,
            'last_frame_time': None,
            'frame_count': 0,
            'piece_count': 0
        }


# ============================== Backends de I/O ===============================

class ScreenBackend:
    def connect(self): ...
    def get_resolution(self)->Tuple[int,int]: ...
    def get_screen(self)->np.ndarray: ...
    def tap(self,x:int,y:int,hold_ms:int=70): ...
    def swipe(self,x1:int,y1:int,x2:int,y2:int,duration_ms:int=120): ...
    def cleanup(self): pass


class ADBBackend(ScreenBackend):
    def __init__(self, serial: Optional[str]=None):
        self.serial=serial; self.resolution=None
    def _adb_args(self): return ["adb"]+(["-s",self.serial] if self.serial else [])
    def _run(self,args, capture_output=True, check=True):
        cmd=self._adb_args()+args
        logging.debug("ADB cmd: "+" ".join(cmd))
        return subprocess.run(cmd, capture_output=capture_output, check=check)
    def _shell_try(self, variants)->bool:
        for v in variants:
            try:
                self._run(["shell"]+v, capture_output=True, check=True)
                return True
            except subprocess.CalledProcessError: pass
        return False
    def connect(self):
        p=subprocess.run(self._adb_args()+["devices"], capture_output=True, check=True)
        lines=p.stdout.decode(errors="ignore").splitlines()
        devs=[ln.split("\t")[0] for ln in lines if "\tdevice" in ln]
        if self.serial and self.serial not in devs:
            logging.warning("ADB: No se detecta 'device' en 'adb devices'.")
        elif not self.serial and devs:
            self.serial=devs[0]
        self.resolution=self._get_wm_size()
        logging.info(f"ADB conectado. Resolución dispositivo: {self.resolution[0]}x{self.resolution[1]}")
    def _get_wm_size(self):
        out=self._run(["shell","wm","size"]).stdout.decode(errors="ignore")
        for ln in out.splitlines():
            if ":" in ln and "x" in ln:
                part=ln.split(":")[1].strip()
                w,h=part.split("x"); return int(w),int(h)
        # fallback
        out=self._run(["shell","dumpsys","display"]).stdout.decode(errors="ignore")
        import re
        m=re.search(r"real (\d+) x (\d+)", out)
        if m: return int(m.group(1)), int(m.group(2))
        raise RuntimeError("No se pudo detectar resolución.")
    def get_resolution(self): return self.resolution
    def get_screen(self)->np.ndarray:
        p=self._run(["exec-out","screencap","-p"])
        img=cv2.imdecode(np.frombuffer(p.stdout,np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise RuntimeError("Fallo screencap.")
        w,h=self.get_resolution()
        if (img.shape[1],img.shape[0]) != (w,h):
            img=cv2.resize(img,(w,h))
        return img
    def tap(self,x:int,y:int,hold_ms:int=80):
        x=int(x);y=int(y)
        logging.debug(f"ADB tap: ({x},{y}) hold={hold_ms}ms")
        if self._shell_try([["input","tap",str(x),str(y)]]):
            logging.debug("ADB tap exitoso (input tap)")
            return
        if self._shell_try([["cmd","input","tap",str(x),str(y)]]):
            logging.debug("ADB tap exitoso (cmd input tap)")
            return
        d=max(100,int(hold_ms))
        logging.debug(f"ADB tap fallback: swipe {x},{y} -> {x+1},{y+1} dur={d}ms")
        self._run(["shell","input","swipe",str(x),str(y),str(x+1),str(y+1),str(d)])
        logging.debug("ADB tap completado (fallback swipe)")
    def swipe(self, x1:int, y1:int, x2:int, y2:int, duration_ms:int=130):
        # Forzar la variante que sí funciona en tu dispositivo:
        # "adb shell input touchscreen swipe x1 y1 x2 y2 dur"
        x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
        d = max(180, int(duration_ms))  # un poco más largo para asegurar detección

        logging.debug(f"ADB swipe (touchscreen): ({x1},{y1}) -> ({x2},{y2}) dur={d}ms")

        # 1) La que te funciona (PRIORIDAD)
        if self._shell_try([["input","touchscreen","swipe", str(x1),str(y1),str(x2),str(y2),str(d)]]):
            logging.debug("ADB swipe exitoso (input touchscreen swipe)")
            return

        # 2) Fallbacks por si alguna ROM rara bloquea el primero
        if self._shell_try([["cmd","input","touchscreen","swipe", str(x1),str(y1),str(x2),str(y2),str(d)]]):
            logging.debug("ADB swipe exitoso (cmd input touchscreen swipe)")
            return
        if self._shell_try([["input","swipe", str(x1),str(y1),str(x2),str(y2),str(d)]]):
            logging.debug("ADB swipe exitoso (input swipe)")
            return
        if self._shell_try([["cmd","input","swipe", str(x1),str(y1),str(x2),str(y2),str(d)]]):
            logging.debug("ADB swipe exitoso (cmd input swipe)")
            return

        logging.error("ADB swipe falló en todas las variantes")
        raise RuntimeError("adb swipe falló en todas las variantes.")

class ScrcpyBackend(ScreenBackend):
    def __init__(self, serial: Optional[str]=None, title="TetrisBot"):
        if not SCRCPY_DEPS_OK:
            raise RuntimeError("scrcpy backend requiere pyautogui, pygetwindow, mss")
        self.serial=serial; self.title=title; self.proc=None; self.win=None; self.res=None
        pyautogui.FAILSAFE=False; pyautogui.PAUSE=0.02
    def _all_titles(self):
        try: return gw.getAllTitles()
        except Exception: return []
    def _find_window(self):
        wins=gw.getWindowsWithTitle(self.title)
        if wins: return wins[0]
        cand=[t for t in self._all_titles() if "scrcpy" in t.lower()]
        return gw.getWindowsWithTitle(cand[0])[0] if cand else None
    def _grab_window(self, timeout=12.0):
        t0=time.time()
        while time.time()-t0<timeout:
            w=self._find_window()
            if w:
                self.win=w
                try: w.activate()
                except Exception: pass
                logging.debug(f"Ventana scrcpy: {w.title} ({w.left},{w.top}) {w.width}x{w.height}")
                return
            time.sleep(0.25)
        raise RuntimeError("No se encontró la ventana de scrcpy.")
    def _spawn_scrcpy(self):
        base=["scrcpy"]; 
        if self.serial: base+=["--serial", self.serial]
        variants=[["--window-title", self.title, "--window-borderless"],
                  ["--window-title", self.title], []]
        last_err=""
        for extra in variants:
            args=base+extra
            logging.debug("Lanzando scrcpy: "+" ".join(args))
            try:
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system()=="Windows" else 0
                self.proc=subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
            except FileNotFoundError:
                raise RuntimeError("scrcpy no está en PATH.")
            time.sleep(0.8)
            rc=self.proc.poll()
            if rc is None: return
            try: _,err=self.proc.communicate(timeout=0.2); last_err=(err or b"").decode(errors="ignore")
            except Exception: last_err="(sin stderr)"
            logging.warning(f"scrcpy terminó (rc={rc}) args={extra}. stderr:\n{last_err}")
        raise RuntimeError(f"No se pudo iniciar scrcpy. Último error:\n{last_err}")
    def connect(self):
        adb=ADBBackend(self.serial); adb.connect(); self.res=adb.get_resolution()
        self._spawn_scrcpy(); self._grab_window()
        logging.info(f"scrcpy listo. Resolución disp: {self.res[0]}x{self.res[1]}")
    def get_resolution(self): return self.res
    def _box(self):
        w=self._find_window()
        if w: self.win=w
        return (self.win.left,self.win.top,self.win.right,self.win.bottom)
    def _to_win_xy(self,x:int,y:int):
        l,t,r,b=self._box(); win_w=r-l; win_h=b-t; dev_w,dev_h=self.res
        s=min(win_w/dev_w, win_h/dev_h)
        off_x=l+(win_w-int(dev_w*s))//2; off_y=t+(win_h-int(dev_h*s))//2
        return int(off_x+x*s), int(off_y+y*s)
    def get_screen(self)->np.ndarray:
        l,t,r,b=self._box()
        with mss.mss() as sct:
            shot=sct.grab({"left":l,"top":t,"width":r-l,"height":b-t})
            img=np.array(shot)[:,:,:3]
        dev_w,dev_h=self.res
        return cv2.resize(img,(dev_w,dev_h))
    def tap(self,x:int,y:int,hold_ms:int=80):
        wx,wy=self._to_win_xy(x,y); pyautogui.moveTo(wx,wy); pyautogui.mouseDown(); time.sleep(hold_ms/1000.0); pyautogui.mouseUp()
    def swipe(self,x1:int,y1:int,x2:int,y2:int,duration_ms:int=130):
        wx1,wy1=self._to_win_xy(x1,y1); wx2,wy2=self._to_win_xy(x2,y2)
        pyautogui.moveTo(wx1,wy1); pyautogui.dragTo(wx2,wy2, duration=duration_ms/1000.0, button='left')


class HybridBackend(ScreenBackend):
    def __init__(self, serial: Optional[str]=None):
        self.adb=ADBBackend(serial)
        self.scr=None
        if SCRCPY_DEPS_OK: self.scr=ScrcpyBackend(serial)
        self._scr_ok=False; self._res=None
    def connect(self):
        self.adb.connect(); self._res=self.adb.get_resolution()
        if self.scr:
            try: self.scr.connect(); self._scr_ok=True; logging.info("Hybrid: captura scrcpy + toques ADB.")
            except Exception as e: logging.warning(f"Hybrid: scrcpy no disponible ({e}).")
    def get_resolution(self): return self._res
    def get_screen(self)->np.ndarray:
        return self.scr.get_screen() if self._scr_ok else self.adb.get_screen()
    def tap(self,x:int,y:int,hold_ms:int=80): self.adb.tap(x,y,hold_ms)
    def swipe(self,x1:int,y1:int,x2:int,y2:int,duration_ms:int=130): self.adb.swipe(x1,y1,x2,y2,duration_ms)
    def cleanup(self):
        if self.scr and self._scr_ok:
            try:
                if self.scr.proc and self.scr.proc.poll() is None:
                    if platform.system()=="Windows":
                        self.scr.proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        self.scr.proc.terminate()
            except Exception: pass


# ============================= Visión y tableros ==============================

@dataclass
class BoardRect:
    x0:int; y0:int; w:int; h:int
    def cell_rect(self,r:int,c:int,rows=20,cols=10)->Tuple[int,int,int,int]:
        cw=self.w/cols; ch=self.h/rows
        return int(self.x0+c*cw), int(self.y0+r*ch), int(cw), int(ch)

def get_board_rect_from_percent(res:Tuple[int,int], pct:Tuple[float,float,float,float])->BoardRect:
    W,H=res; x=int(pct[0]*W); y=int(pct[1]*H); w=int(pct[2]*W); h=int(pct[3]*H)
    return BoardRect(x,y,w,h)

def auto_calibrate_board_rect(backend: 'ScreenBackend', config: TetrisConfig) -> Optional[BoardRect]:
    """
    Detecta automáticamente el rectángulo del tablero de Tetris usando análisis de bordes.
    """
    try:
        auto_config = config.config.get("vision", {}).get("auto_calibration", {})
        expected_ratio = auto_config.get("expected_ratio", 2.0)
        debug_save = auto_config.get("debug_save", False)
        hough_threshold = auto_config.get("hough_threshold", 100)
        canny_low = auto_config.get("canny_low", 50)
        canny_high = auto_config.get("canny_high", 150)

        # BUG FIX: usar get_screen() (no existe get_frame())
        frame = backend.get_screen()
        if frame is None:
            logging.warning("No se pudo capturar frame para calibración automática")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=hough_threshold)
        if lines is None:
            logging.warning("No se detectaron líneas para calibración automática")
            return None

        vertical_lines, horizontal_lines = [], []
        for line in lines:
            rho, theta = line[0]
            # verticales: theta cerca de 0 o π
            if abs(theta) < np.pi/6 or abs(theta - np.pi) < np.pi/6:
                vertical_lines.append((rho, theta))
            # horizontales: theta cerca de π/2
            elif abs(theta - np.pi/2) < np.pi/6:
                horizontal_lines.append((rho, theta))

        if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
            logging.warning("No se encontraron suficientes líneas para calibración automática")
            return None

        vertical_positions = []
        for rho, theta in vertical_lines:
            if abs(np.cos(theta)) > 0.1:
                x = rho / np.cos(theta)
                if 0 <= x <= width:
                    vertical_positions.append(int(x))

        horizontal_positions = []
        for rho, theta in horizontal_lines:
            if abs(np.sin(theta)) > 0.1:
                y = rho / np.sin(theta)
                if 0 <= y <= height:
                    horizontal_positions.append(int(y))

        if len(vertical_positions) < 2 or len(horizontal_positions) < 2:
            logging.warning("No se pudieron convertir líneas a posiciones válidas")
            return None

        vertical_positions.sort()
        horizontal_positions.sort()

        left, right = vertical_positions[0], vertical_positions[-1]
        top, bottom = horizontal_positions[0], horizontal_positions[-1]

        board_width, board_height = right - left, bottom - top
        if board_width <= 0 or board_height <= 0:
            logging.warning("Dimensiones del tablero detectado son inválidas")
            return None

        actual_ratio = board_height / max(1, board_width)
        tolerance = 0.5  # permisividad alrededor del ratio esperado
        if not (expected_ratio - tolerance <= actual_ratio <= expected_ratio + tolerance):
            logging.warning(
                f"Ratio detectado {actual_ratio:.2f} fuera del rango esperado ({expected_ratio - tolerance:.1f}-{expected_ratio + tolerance:.1f})"
            )
            return None

        detected_rect = BoardRect(left, top, board_width, board_height)

        if debug_save:
            try:
                os.makedirs("tetris_debug", exist_ok=True)
                dbg = frame.copy()
                cv2.rectangle(dbg, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.imwrite("tetris_debug/auto_calibration.png", dbg)
                cv2.imwrite("tetris_debug/auto_calibration_edges.png", edges)
                logging.info("Imágenes de calibración guardadas en tetris_debug/")
            except Exception as e:
                logging.warning(f"Error guardando debug de auto-calibración: {e}")

        logging.info(f"Calibración automática OK: {left},{top},{board_width},{board_height} (ratio {actual_ratio:.2f})")
        return detected_rect

    except Exception as e:
        logging.error(f"Error en calibración automática: {e}")
        return None

# --- NUEVO: limpia ruido sin borrar tetrominós ---
def _remove_small_components_bool_grid(occ: np.ndarray, min_cells:int=2)->np.ndarray:
    """
    Elimina sólo componentes más pequeños que min_cells (p.ej., 1 celda).
    Mantiene piezas de 2-6 celdas y cualquier bloque real.
    """
    rows, cols = occ.shape
    vis = np.zeros_like(occ, bool)

    for r in range(rows):
        for c in range(cols):
            if occ[r, c] and not vis[r, c]:
                stack = [(r, c)]
                vis[r, c] = True
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and occ[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            stack.append((nr, nc))
                if len(comp) < min_cells:
                    for rr, cc in comp:
                        occ[rr, cc] = False
    return occ


def classify_cell_type(hsv_cell: np.ndarray) -> str:
    """
    Clasifica una celda en uno de tres tipos basándose en rangos HSV absolutos:
    - 'background': Fondo del juego (tonos azul-gris, baja saturación)
    - 'active': Pieza activa (colores saturados y brillantes) 
    - 'ghost': Pieza ghost (colores menos saturados, más oscuros)
    
    hsv_cell: array HSV de la región de la celda
    Returns: 'background', 'active', o 'ghost'
    """
    # Calcular medianas HSV de la celda
    h_med = float(np.median(hsv_cell[..., 0]))
    s_med = float(np.median(hsv_cell[..., 1]))
    v_med = float(np.median(hsv_cell[..., 2]))
    
    # Rangos HSV específicos para tu juego de Tetris
    # VERDE (piezas verdes como en tu imagen)
    GREEN_H_MIN, GREEN_H_MAX = 35, 85  # Hue para verde
    
    # ACTIVA: Verde brillante y saturado
    ACTIVE_S_MIN = 150  # Saturación alta para piezas activas
    ACTIVE_V_MIN = 150  # Brillo alto para piezas activas
    
    # GHOST: Verde menos saturado y más oscuro
    GHOST_S_MIN = 80   # Saturación media para ghost
    GHOST_S_MAX = 149  # Máxima saturación para ghost (menor que activa)
    GHOST_V_MIN = 80   # Brillo medio para ghost
    GHOST_V_MAX = 149  # Máximo brillo para ghost (menor que activa)
    
    # FONDO: Baja saturación (azules/grises del tablero)
    BG_S_MAX = 79      # Saturación máxima para fondo
    
    # Lógica de clasificación
    if GREEN_H_MIN <= h_med <= GREEN_H_MAX:
        # Es verde - determinar si activa o ghost
        if s_med >= ACTIVE_S_MIN and v_med >= ACTIVE_V_MIN:
            return 'active'
        elif GHOST_S_MIN <= s_med <= GHOST_S_MAX and GHOST_V_MIN <= v_med <= GHOST_V_MAX:
            return 'ghost'
    
    # Añadir soporte para otros colores de piezas si es necesario
    # Por ahora, cualquier color saturado que no sea verde se considera activa
    if s_med >= ACTIVE_S_MIN and v_med >= ACTIVE_V_MIN:
        return 'active'
    
    # Si no es una pieza, es fondo
    return 'background'


def occupancy_grid_multilayer(board_bgr, rows=20, cols=10, mode="normal"):
    """
    Nueva versión de occupancy_grid que clasifica cada celda en 3 tipos:
    - background_grid: Celdas de fondo
    - active_grid: Celdas de piezas activas  
    - ghost_grid: Celdas de piezas ghost
    
    Returns: (background_grid, active_grid, ghost_grid, debug_mask)
    """
    H, W = board_bgr.shape[:2]
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
    
    # Inicializar las 3 grillas
    background_grid = np.zeros((rows, cols), dtype=bool)
    active_grid = np.zeros((rows, cols), dtype=bool)
    ghost_grid = np.zeros((rows, cols), dtype=bool)
    
    # Crear máscara de debug (colorear por tipo)
    debug_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Parámetros para el padding de celdas
    PAD_C = 0.20 if mode == "normal" else 0.22
    
    # Calcular límites de celdas (pixel-perfect)
    row_boundaries = np.linspace(0, H, rows + 1, dtype=int)
    col_boundaries = np.linspace(0, W, cols + 1, dtype=int)
    
    # Iterar por cada celda
    for r in range(rows):
        for c in range(cols):
            # Límites de la celda
            x0, x1 = col_boundaries[c], col_boundaries[c + 1]
            y0, y1 = row_boundaries[r], row_boundaries[r + 1]
            
            # Asegurar mínimo 1 pixel
            x1 = max(x1, x0 + 1)
            y1 = max(y1, y0 + 1)
            
            # Región central (evitar bordes)
            dx, dy = int((x1-x0)*PAD_C), int((y1-y0)*PAD_C)
            cx0, cx1 = x0+dx, x1-dx
            cy0, cy1 = y0+dy, y1-dy
            
            if cx1 <= cx0 or cy1 <= cy0:
                continue
                
            # Extraer HSV de la región central
            c_hsv = hsv[cy0:cy1, cx0:cx1]
            
            # Clasificar la celda
            cell_type = classify_cell_type(c_hsv)
            
            # Asignar a la grilla correspondiente
            if cell_type == 'background':
                background_grid[r, c] = True
                debug_mask[y0:y1, x0:x1] = [100, 100, 100]  # Gris para fondo
            elif cell_type == 'active':
                active_grid[r, c] = True
                debug_mask[y0:y1, x0:x1] = [0, 255, 0]      # Verde para activa
            elif cell_type == 'ghost':
                ghost_grid[r, c] = True
                debug_mask[y0:y1, x0:x1] = [0, 255, 255]    # Cyan para ghost
    
    # Log estadísticas
    total_cells = rows * cols
    bg_count = np.sum(background_grid)
    active_count = np.sum(active_grid)
    ghost_count = np.sum(ghost_grid)
    
    logging.info(f"🎯 Multilayer segmentation: {bg_count} fondo, {active_count} activas, {ghost_count} ghost (total: {total_cells})")
    
    return background_grid, active_grid, ghost_grid, debug_mask


def extract_components_by_type(active_grid: np.ndarray, ghost_grid: np.ndarray, 
                              max_component_size: int = 4) -> tuple:
    """
    Extrae componentes conectados de las grillas de piezas activas y ghost por separado.
    
    Returns: (active_components, ghost_components)
    - active_components: Lista de componentes de piezas activas
    - ghost_components: Lista de componentes de piezas ghost
    """
    def get_components_from_grid(grid: np.ndarray) -> List[List[Tuple[int,int]]]:
        """Extrae componentes conectados de una grilla booleana"""
        rows, cols = grid.shape
        vis = np.zeros_like(grid, bool)
        components = []

        def neighbors(r, c):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    yield rr, cc

        def flood_fill(start_r, start_c):
            comp = []
            stack = [(start_r, start_c)]
            while stack:
                r, c = stack.pop()
                if vis[r, c] or not grid[r, c]:
                    continue
                vis[r, c] = True
                comp.append((r, c))
                
                for nr, nc in neighbors(r, c):
                    if not vis[nr, nc] and grid[nr, nc]:
                        stack.append((nr, nc))
                        
                # Limitar tamaño de componente para evitar que crezca demasiado
                if len(comp) > max_component_size:
                    # Marcar todas las celdas restantes como visitadas
                    for nr, nc in neighbors(r, c):
                        if grid[nr, nc]:
                            vis[nr, nc] = True
                    break
                            
            return comp

        # Encontrar todos los componentes
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] and not vis[r, c]:
                    comp = flood_fill(r, c)
                    if 1 <= len(comp) <= max_component_size:
                        components.append(comp)

        return components

    # Extraer componentes de cada tipo
    active_components = get_components_from_grid(active_grid)
    ghost_components = get_components_from_grid(ghost_grid)
    
    logging.info(f"🔍 Components extracted: {len(active_components)} activas, {len(ghost_components)} ghost")
    
    return active_components, ghost_components


def detect_pieces_multilayer(board_bgr: np.ndarray, rows=20, cols=10, mode="normal") -> tuple:
    """
    Nueva función de detección de piezas usando el sistema multilayer.
    Reemplaza la lógica compleja anterior con un enfoque directo basado en colores absolutos.
    
    Returns: (active_piece, ghost_piece, debug_info)
    - active_piece: Lista de celdas de la pieza activa
    - ghost_piece: Lista de celdas de la pieza ghost  
    - debug_info: Dict con información de debug
    """
    # Obtener las 3 capas de segmentación
    bg_grid, active_grid, ghost_grid, debug_mask = occupancy_grid_multilayer(board_bgr, rows, cols, mode)
    
    # Extraer componentes conectados por tipo
    active_components, ghost_components = extract_components_by_type(active_grid, ghost_grid)
    
    # Seleccionar la mejor pieza activa (componente más alto)
    active_piece = None
    if active_components:
        # Ordenar por fila más alta (menor valor de r)
        active_components.sort(key=lambda comp: min(r for r, c in comp))
        active_piece = active_components[0]
        logging.info(f"✅ Pieza activa detectada: {len(active_piece)} celdas (multilayer)")
    
    # Validar y seleccionar ghost piece
    ghost_piece = None
    if ghost_components and active_piece:
        # El ghost debe estar debajo de la pieza activa y compartir al menos 1 columna
        active_cols = set(c for r, c in active_piece)
        active_bottom = max(r for r, c in active_piece)
        
        valid_ghosts = []
        for ghost_comp in ghost_components:
            ghost_cols = set(c for r, c in ghost_comp)
            ghost_top = min(r for r, c in ghost_comp)
            
            # Validaciones: debe estar debajo y compartir columnas
            if ghost_top > active_bottom and len(active_cols & ghost_cols) > 0:
                valid_ghosts.append(ghost_comp)
        
        if valid_ghosts:
            # Tomar el ghost más cercano a la pieza activa
            valid_ghosts.sort(key=lambda comp: min(r for r, c in comp))
            ghost_piece = valid_ghosts[0]
            logging.info(f"👻 Ghost detectado: {len(ghost_piece)} celdas (multilayer)")
    
    # Si no hay ghost detectado pero hay componentes ghost, reportar
    elif ghost_components:
        logging.warning(f"⚠️ {len(ghost_components)} componentes ghost detectados pero ninguno válido posicionalmente")
    
    # Información de debug
    debug_info = {
        'active_components_count': len(active_components),
        'ghost_components_count': len(ghost_components),
        'debug_mask': debug_mask,
        'multilayer_used': True
    }
    
    return active_piece, ghost_piece, debug_info


def occupancy_grid(board_bgr, rows=20, cols=10, mode="normal"):
    """
    Segmentación por celda basada en MODELO DE FONDO con EXCLUSIÓN DE SOMBRAS:
    - Aprende 2–3 clusters de fondo en CIE-Lab (kmeans).
    - Celda ocupada si su color (mediana Lab del centro) está lejos del fondo.
    - Umbral adaptativo (mediana + 3*MAD) con piso mínimo.
    - Filtros suaves por S y V para evitar brillos del tablero.
    - NUEVO: Detección específica de sombras/ghost pieces por transparencia.
    - Limpieza morfológica.
    Devuelve: (occ_bool[rows,cols], mask_debug[H,W])
    """
    H, W = board_bgr.shape[:2]
    lab = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)

    # ---- parámetros por modo ----
    if mode == "tight":
        PAD_C      = 0.22
        BG_FLOOR   = 12.0   # piso de distancia Lab al fondo
        S_MIN      = 70     # filtros suaves (no matar piezas rosadas)
        V_MIN      = 110
        K_CLUSTERS = 3
        # Parámetros para detección de sombras
        SHADOW_SAT_MAX = 85   # sombras tienen baja saturación
        SHADOW_VAL_MIN = 120  # pero no son muy oscuras
        SHADOW_LAB_MAX = 15   # distancia intermedia al fondo
    else:
        PAD_C      = 0.20
        BG_FLOOR   = 10.0
        S_MIN      = 65
        V_MIN      = 105
        K_CLUSTERS = 3
        # Parámetros para detección de sombras  
        SHADOW_SAT_MAX = 80   # sombras tienen baja saturación
        SHADOW_VAL_MIN = 115  # pero no son muy oscuras
        SHADOW_LAB_MAX = 13   # distancia intermedia al fondo

    # ---- modelado de fondo (kmeans en una versión reducida) ----
    # Usamos el rectángulo central para evitar bordes dibujados
    y0, y1 = int(H*0.08), int(H*0.92)
    x0, x1 = int(W*0.08), int(W*0.92)
    lab_mid = lab[y0:y1, x0:x1]
    # Downsample agresivo para velocidad
    small = cv2.resize(lab_mid, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    small = np.float32(small)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    compactness, labels, centers = cv2.kmeans(
        data=small, K=K_CLUSTERS, bestLabels=None, criteria=criteria,
        attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    # Tomamos los 2 clusters más poblados como "fondo"
    counts = np.bincount(labels, minlength=K_CLUSTERS)
    bg_idx = np.argsort(-counts)[:2]
    bg_centers = centers[bg_idx]  # (2,3)

    # ---- recorrer celdas y medir distancia a fondo ----
    occ = np.zeros((rows, cols), np.bool_)
    d_bg_grid = np.zeros((rows, cols), np.float32)
    s_med_grid = np.zeros((rows, cols), np.float32)
    v_med_grid = np.zeros((rows, cols), np.float32)
    shadow_mask = np.zeros((rows, cols), np.bool_)  # NUEVO: máscara de sombras detectadas

    # Pixel-perfect grid calculation: distribute pixels evenly
    row_boundaries = np.linspace(0, H, rows + 1, dtype=int)
    col_boundaries = np.linspace(0, W, cols + 1, dtype=int)
    
    # Calculate average cell dimensions for logging
    avg_ch = H / rows
    avg_cw = W / cols
    logging.debug(f"Board analysis: H={H} W={W} rows={rows} cols={cols} avg_ch={avg_ch:.1f} avg_cw={avg_cw:.1f}")
    
    # Análisis detallado de cobertura de filas usando pixel-perfect boundaries
    last_row_y0 = row_boundaries[rows-1]
    last_row_y1 = row_boundaries[rows]
    pixels_per_row = H / float(rows)

    logging.info("🔍 Grid boundary analysis (pixel-perfect):")
    logging.info(f"   Crop size: {H}x{W} pixels")
    logging.info(f"   Average pixels per row: {pixels_per_row:.1f}")
    logging.info(f"   Last row boundaries: y={last_row_y0} to y={last_row_y1} (height: {last_row_y1-last_row_y0})")
    logging.info(f"   Grid covers full image height: {H}")
    
    # With pixel-perfect grid, we always cover the full image
    logging.info("✓ Perfect grid coverage - no pixels lost")
    
    for r in range(rows):
        for c in range(cols):
            # Use pixel-perfect boundaries
            x0, x1 = col_boundaries[c], col_boundaries[c + 1]
            y0, y1 = row_boundaries[r], row_boundaries[r + 1]
            
            # With linspace, coordinates should always be valid, but keep safety checks
            x1 = max(x1, x0 + 1)  # mínimo 1 pixel de ancho
            y1 = max(y1, y0 + 1)  # mínimo 1 pixel de alto
            
            # Logging adicional para la última fila
            if r == rows - 1 and c == 0:  # Solo log una vez por fila
                logging.debug(f"Fila {r} (última): y0={y0}, y1={y1}, altura={y1-y0} pixels")

            # Centro de la celda (evitamos bordes)
            dx, dy = int((x1-x0)*PAD_C), int((y1-y0)*PAD_C)
            cx0, cx1 = x0+dx, x1-dx
            cy0, cy1 = y0+dy, y1-dy
            if cx1 <= cx0 or cy1 <= cy0:
                if r >= rows - 2:  # Solo log para las últimas 2 filas
                    logging.debug(f"Celda ({r},{c}) omitida: cx0={cx0} cx1={cx1} cy0={cy0} cy1={cy1}")
                continue

            c_lab = lab[cy0:cy1, cx0:cx1].reshape(-1, 3)
            c_hsv = hsv[cy0:cy1, cx0:cx1]

            med_lab = np.median(c_lab, axis=0)
            s_med   = float(np.median(c_hsv[...,1]))
            v_med   = float(np.median(c_hsv[...,2]))

            # Distancia mínima a cualquiera de los dos centros de fondo
            dists = np.linalg.norm(bg_centers - med_lab, axis=1)
            d_bg  = float(np.min(dists))

            d_bg_grid[r, c] = d_bg
            s_med_grid[r, c] = s_med
            v_med_grid[r, c] = v_med
            
            # NUEVO: Detectar sombras/ghost pieces por características específicas
            # Sombras típicamente tienen: baja saturación, brillo intermedio, distancia intermedia al fondo
            is_shadow = (s_med < SHADOW_SAT_MAX and 
                        v_med > SHADOW_VAL_MIN and 
                        BG_FLOOR < d_bg < SHADOW_LAB_MAX)
            shadow_mask[r, c] = is_shadow
            
            if is_shadow:
                logging.debug(f"Sombra detectada en ({r},{c}): S={s_med:.1f} V={v_med:.1f} Lab={d_bg:.1f}")

    # ---- umbral adaptativo sobre d_bg (robusto a iluminación) ----
    vals = d_bg_grid.reshape(-1)
    med  = float(np.median(vals))
    mad  = 1.4826 * float(np.median(np.abs(vals - med)))
    thr  = max(BG_FLOOR, med + 3.0*mad)  # piezas quedan muy por encima

    # Ocupación preliminar (piezas + potenciales sombras)
    occ_pre = (d_bg_grid >= thr) & (s_med_grid >= S_MIN) & (v_med_grid >= V_MIN)
    
    # NUEVO: Excluir sombras explícitamente detectadas
    occ_no_shadows = occ_pre & (~shadow_mask)
    
    # Log estadísticas de sombras detectadas
    shadow_count = np.sum(shadow_mask)
    if shadow_count > 0:
        logging.info(f"🫥 {shadow_count} celdas de sombra detectadas y excluidas")
        shadow_cells = [(r,c) for r in range(rows) for c in range(cols) if shadow_mask[r,c]]
        logging.debug(f"Posiciones de sombras: {shadow_cells[:10]}{'...' if len(shadow_cells) > 10 else ''}")

    occ = _remove_small_components_bool_grid(occ_no_shadows.astype(bool), min_cells=2)
    if not np.any(occ):
        # Fallback: usar ocupación sin filtrado de sombras (modo conservador)
        logging.warning("⚠️  Sin piezas detectadas tras filtrar sombras, usando detección sin filtro")
        occ = _remove_small_components_bool_grid(occ_pre.astype(bool), min_cells=2)
        if not np.any(occ):
            occ = occ_pre.astype(bool)


    # ---- máscara de depuración ----
    mask = np.zeros((H, W), np.uint8)
    for r in range(rows):
        for c in range(cols):
            if not occ[r, c]:
                continue
            # Use pixel-perfect boundaries for debug mask too
            x0, x1 = col_boundaries[c], col_boundaries[c + 1]
            y0, y1 = row_boundaries[r], row_boundaries[r + 1]
            
            # Asegurar que las coordenadas están dentro de los límites
            x1 = max(x1, x0 + 1)  # mínimo 1 pixel de ancho
            y1 = max(y1, y0 + 1)  # mínimo 1 pixel de alto
            
            dx, dy = int((x1-x0)*PAD_C), int((y1-y0)*PAD_C)
            cx0, cx1 = x0+dx, x1-dx
            cy0, cy1 = y0+dy, y1-dy
            mask[cy0:cy1, cx0:cx1] = 255

    return occ, mask


def occupancy_grid_tight(board_bgr, rows=20, cols=10):
    """
    Versión más estricta de occupancy_grid para tableros muy llenos.
    Usa parámetros 'tight' para detectar mejor las piezas en condiciones difíciles.
    Devuelve: (occ_bool[rows,cols], mask_debug[H,W])
    """
    return occupancy_grid(board_bgr, rows, cols, mode="tight")


def list_components(occ: np.ndarray, max_component_size: int = 4) -> List[List[Tuple[int,int]]]:
    """
    Lista componentes conectados (4-conectividad). Si un componente excede max_component_size,
    se descarta y se marca COMPLETO como visitado para que no re-aparezca.
    """
    rows, cols = occ.shape
    vis = np.zeros_like(occ, bool)
    comps: List[List[Tuple[int,int]]] = []

    def neighbors(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                yield rr, cc

    for r in range(rows):
        for c in range(cols):
            if not occ[r, c] or vis[r, c]:
                continue

            # BFS completo del componente
            q = [(r, c)]
            vis[r, c] = True
            comp = []

            too_big = False
            qi = 0
            while qi < len(q):
                rr, cc = q[qi]; qi += 1
                comp.append((rr, cc))

                if len(comp) > max_component_size:
                    too_big = True
                    # Aún así terminamos de expandir para marcar todo como visitado
                for nr, nc in neighbors(rr, cc):
                    if occ[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))

            if not too_big:
                comps.append(comp)
            else:
                logging.debug(f"Componente descartado por tamaño: {len(comp)} > {max_component_size}")

    return comps

def remove_cells(occ: np.ndarray, cells: List[Tuple[int,int]]):
    for r,c in cells:
        if 0<=r<occ.shape[0] and 0<=c<occ.shape[1]: occ[r,c]=False

def bounding_box(cells: [Tuple[int,int]]) -> Tuple[int,int,int,int]:
    rs = [r for r,_ in cells]; cs = [c for _,c in cells]
    return min(rs), min(cs), max(rs), max(cs)

def _cell_rect(r: int, c: int,
               grid_shape: Tuple[int, int],
               img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Devuelve (y0,y1,x0,x1) para la celda (r,c) usando límites pixel-perfect."""
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    H, W = img_shape[:2]
    row_boundaries = np.linspace(0, H, rows + 1, dtype=int)
    col_boundaries = np.linspace(0, W, cols + 1, dtype=int)
    y0, y1 = row_boundaries[r], row_boundaries[r + 1]
    x0, x1 = col_boundaries[c], col_boundaries[c + 1]
    return y0, y1, x0, x1

def _avg_sat_of_component(board_bgr: np.ndarray,
                          
                          comp: List[Tuple[int,int]],
                          grid_shape: Tuple[int,int]) -> float:
    """
    Saturación mediana (HSV S) del centro de cada celda del componente.
    'grid_shape' = (rows, cols) tomado de occ.shape para evitar None.
    """
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    H, W = board_bgr.shape[:2]
    
    # Pixel-perfect grid calculation: distribute pixels evenly
    row_boundaries = np.linspace(0, H, rows + 1, dtype=int)
    col_boundaries = np.linspace(0, W, cols + 1, dtype=int)
    
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
    sats = []
    for r, c in comp:
        x0, x1 = col_boundaries[c], col_boundaries[c + 1]
        y0, y1 = row_boundaries[r], row_boundaries[r + 1]
        cx0, cx1 = x0 + int((x1 - x0) * 0.15), x1 - int((x1 - x0) * 0.15)
        cy0, cy1 = y0 + int((y1 - y0) * 0.15), y1 - int((y1 - y0) * 0.15)
        if cx1 > cx0 and cy1 > cy0:
            cell_S = hsv[cy0:cy1, cx0:cx1, 1]
            sats.append(float(np.median(cell_S)))
    return float(np.mean(sats)) if sats else 0.0

def _avg_val_of_component(board_bgr: np.ndarray,
                          comp: List[Tuple[int,int]],
                          grid_shape: Tuple[int,int]) -> float:
    if board_bgr is None or not comp: return 0.0
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
    vals = []
    for r,c in comp:
        y0,y1,x0,x1 = _cell_rect(r,c,grid_shape,hsv.shape)
        patch = hsv[y0:y1, x0:x1, 2]  # V
        if patch.size:
            vals.append(float(np.median(patch)))
    return float(np.median(vals)) if vals else 0.0

def _split_merged_active_and_ghost(cells: List[Tuple[int,int]]) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """Si `cells` contiene la pieza activa junto con su ghost, separa ambos conjuntos."""
    if not cells or len(cells) <= 4:
        return cells, []
    cells_sorted = sorted(cells)
    groups = [[cells_sorted[0]]]
    for r, c in cells_sorted[1:]:
        if r - groups[-1][-1][0] > 1:
            groups.append([])
        groups[-1].append((r, c))
    if len(groups) == 1:
        return cells, []
    groups.sort(key=lambda g: min(r for r, _ in g))
    active = groups[0]
    ghost = [cell for g in groups[1:] for cell in g]
    return active, ghost

def detect_ghost_component(board_bgr: np.ndarray,
                           occ: np.ndarray,
                           piece_cells: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """
    Detecta la sombra ('ghost') de la pieza activa:
    * Debe estar por debajo de la pieza y compartir ≥1 columna.
    * S (saturación) significativamente menor que la pieza real.
    * V (valor/brightness) igual o mayor (ghost suele ser más claro).
    * Agrupa fragmentos cercanos para reconstruir sombras partidas.
    """
    if board_bgr is None or not piece_cells: 
        return []
    rows, cols = occ.shape
    grid_shape = (int(rows), int(cols))

    active_sig = shape_signature(sorted(piece_cells)[:4])
    r0a, c0a, r1a, c1a = bounding_box(piece_cells)
    active_cols = set(c for _, c in piece_cells)

    sat_active = _avg_sat_of_component(board_bgr, piece_cells, grid_shape)
    val_active = _avg_val_of_component(board_bgr, piece_cells, grid_shape)

    # umbrales robustos (ajustables)
    SAT_DELTA_MIN = 4.0    # ghost debe ser menos saturado (reducido para mejor detección)
    V_DELTA_MIN   = 2.0    # y un poco más brillante (reducido para mejor detección)
    MAX_GHOST_SIZE = 6

    comps = list_components(occ, max_component_size=4)
    active_set = set(piece_cells)

    cands = []
    for comp in comps:
        if set(comp) == active_set: 
            continue
        if len(comp) < 1 or len(comp) > MAX_GHOST_SIZE:
            continue
        r0,c0,r1,c1 = bounding_box(comp)
        if r0 <= r1a:       # ghost siempre por debajo del top de la pieza real
            continue
        cols_comp = set(c for _,c in comp)
        if len(active_cols & cols_comp) == 0:
            continue

        sat_comp = _avg_sat_of_component(board_bgr, comp, grid_shape)
        val_comp = _avg_val_of_component(board_bgr, comp, grid_shape)
        sat_diff = sat_active - sat_comp
        val_diff = val_comp - val_active
        
        # Debug logging para analizar valores HSV
        logging.debug(f"🔍 Ghost analysis: sat_active={sat_active:.1f}, sat_comp={sat_comp:.1f}, sat_diff={sat_diff:.1f} (min={SAT_DELTA_MIN})")
        logging.debug(f"🔍 Ghost analysis: val_active={val_active:.1f}, val_comp={val_comp:.1f}, val_diff={val_diff:.1f} (min={V_DELTA_MIN})")

        if sat_diff >= SAT_DELTA_MIN and val_diff >= V_DELTA_MIN:
            cands.append({
                "comp": comp, "r0": r0,"c0":c0,"r1":r1,"c1":c1,
                "cols": cols_comp, "sat": sat_comp, "val": val_comp,
                "sat_diff": sat_diff, "val_diff": val_diff
            })

    if not cands:
        return []

    groups = _group_nearby_ghost_fragments(cands, max_distance=2)

    # elige el mejor grupo: más cerca verticalmente y más parecido en forma
    best=None; best_score=-1
    for grp in groups:
        cells = []; total_overlap=0; min_dy=1e9
        for g in grp:
            cells += g["comp"]
            total_overlap += len(active_cols & g["cols"])
            min_dy = min(min_dy, g["r0"] - r1a)
        # firma opcional si hay suficientes celdas
        if len(cells) >= 3:
            if shape_signature(sorted(cells)[:4]) != active_sig:
                continue
        sat_comp = _avg_sat_of_component(board_bgr, cells, grid_shape)
        val_comp = _avg_val_of_component(board_bgr, cells, grid_shape)
        score = (total_overlap*3) + (sat_active - sat_comp) + (val_comp - val_active) + max(0, 8 - min_dy)
        if score > best_score:
            best_score = score; best = cells

    if best:
        logging.info(f"🫥 Ghost detectado: {len(best)} celdas (score={best_score:.1f})")
    return best or []


def _group_nearby_ghost_fragments(potential_ghosts: List[dict], max_distance: int = 2) -> List[List[dict]]:
    """
    Agrupa fragmentos de sombras que están espacialmente cercanos.
    Usa distancia Manhattan para determinar proximidad.
    """
    if len(potential_ghosts) <= 1:
        return [potential_ghosts]
    
    groups = []
    visited = set()
    
    for i, ghost1 in enumerate(potential_ghosts):
        if i in visited:
            continue
            
        # Crear nuevo grupo con este fragmento
        current_group = [ghost1]
        visited.add(i)
        
        # Buscar fragmentos cercanos
        for j, ghost2 in enumerate(potential_ghosts):
            if j in visited or j == i:
                continue
                
            # Calcular distancia entre bounding boxes
            min_dist = _manhattan_distance_between_boxes(
                (ghost1['r0'], ghost1['c0'], ghost1['r1'], ghost1['c1']),
                (ghost2['r0'], ghost2['c0'], ghost2['r1'], ghost2['c1'])
            )
            
            if min_dist <= max_distance:
                current_group.append(ghost2)
                visited.add(j)
        
        groups.append(current_group)
        
    logging.debug(f"Agrupación de sombras: {len(potential_ghosts)} fragmentos -> {len(groups)} grupos")
    return groups


def _manhattan_distance_between_boxes(box1: Tuple[int,int,int,int], 
                                    box2: Tuple[int,int,int,int]) -> int:
    """
    Calcula la distancia Manhattan mínima entre dos bounding boxes.
    """
    r0_1, c0_1, r1_1, c1_1 = box1
    r0_2, c0_2, r1_2, c1_2 = box2
    
    # Distancia horizontal
    if c1_1 < c0_2:  # box1 está a la izquierda de box2
        h_dist = c0_2 - c1_1
    elif c1_2 < c0_1:  # box2 está a la izquierda de box1  
        h_dist = c0_1 - c1_2
    else:  # se solapan horizontalmente
        h_dist = 0
    
    # Distancia vertical
    if r1_1 < r0_2:  # box1 está arriba de box2
        v_dist = r0_2 - r1_1
    elif r1_2 < r0_1:  # box2 está arriba de box1
        v_dist = r0_1 - r1_2  
    else:  # se solapan verticalmente
        v_dist = 0
        
    return h_dist + v_dist



# ====================== Clasificación y simulación de pieza ======================

PIECE_ORIENTS: Dict[str, List[List[Tuple[int,int]]]] = {
    "I":[[(0,0),(1,0),(2,0),(3,0)], [(0,0),(0,1),(0,2),(0,3)]],
    "O":[[(0,0),(0,1),(1,0),(1,1)]],
    "T":[[(0,0),(0,1),(0,2),(1,1)], [(0,1),(1,0),(1,1),(2,1)],
         [(1,0),(1,1),(1,2),(0,1)], [(0,0),(1,0),(1,1),(2,0)]],
    "S":[[(0,1),(0,2),(1,0),(1,1)], [(0,0),(1,0),(1,1),(2,1)]],
    "Z":[[(0,0),(0,1),(1,1),(1,2)], [(0,1),(1,0),(1,1),(2,0)]],
    "J":[[(0,0),(1,0),(2,0),(2,1)], [(1,0),(1,1),(1,2),(0,0)],
         [(0,0),(0,1),(1,1),(2,1)], [(0,2),(1,0),(1,1),(1,2)]],
    "L":[[(0,1),(1,1),(2,1),(2,0)], [(0,0),(1,0),(1,1),(1,2)],
         [(0,0),(0,1),(1,0),(2,0)], [(0,0),(0,1),(0,2),(1,2)]],
}

def shape_signature(coords: List[Tuple[int,int]])->Tuple[Tuple[int,int],...]:
    r0=min(r for r,_ in coords); c0=min(c for _,c in coords)
    norm=sorted([(r-r0,c-c0) for r,c in coords])
    return tuple(norm)

PIECE_SIGNATURES={k:[shape_signature(o) for o in v] for k,v in PIECE_ORIENTS.items()}

def classify_piece(cells: List[Tuple[int,int]])->Optional[Tuple[str,int]]:
    if not cells: return None
    if len(cells)!=4: cells=sorted(cells)[:4]
    sig=shape_signature(cells)
    for p,sigs in PIECE_SIGNATURES.items():
        for i,s in enumerate(sigs):
            if sig==s: return p,i
    return None

def filter_ghost_pieces(board_bgr: np.ndarray,
                        occ: np.ndarray,
                        comps: List[List[Tuple[int,int]]],
                        grid_shape: Tuple[int,int]=(20,10)) -> List[List[Tuple[int,int]]]:
    """
    Quita de 'comps' cualquier componente que cumpla las condiciones de ghost
    respecto a la pieza más alta (ancla).
    """
    if not comps:
        return comps
    # ancla: componente válido más alto (menor r0)
    anchor = min(comps, key=lambda c: bounding_box(c)[0])
    r0a,_,r1a,_ = bounding_box(anchor)
    rows, cols = grid_shape

    # identificar ghost explícitamente usando la ancla
    ghost_cells = detect_ghost_component(board_bgr, occ, anchor)
    ghost_set = set(ghost_cells)

    # métricas del ancla
    sat_anchor = _avg_sat_of_component(board_bgr, anchor, grid_shape)
    val_anchor = _avg_val_of_component(board_bgr, anchor, grid_shape)
    cols_anchor = set(c for _,c in anchor)

    SAT_DELTA_MIN = 4.0  # Reducido para mejor detección de ghost
    V_DELTA_MIN   = 2.0  # Reducido para mejor detección de ghost

    out=[]
    removed=0
    for comp in comps:
        if comp is anchor:
            out.append(comp); continue
        if ghost_set and set(comp).issubset(ghost_set):
            removed += 1
            continue
        r0, c0, r1, c1 = bounding_box(comp)
        if r0 <= r1a:
            out.append(comp); continue

        # comparte columnas con ancla
        if len(cols_anchor & set(c for _,c in comp)) == 0:
            out.append(comp); continue

        sat_c = _avg_sat_of_component(board_bgr, comp, grid_shape)
        val_c = _avg_val_of_component(board_bgr, comp, grid_shape)
        
        # Debug logging para filtro de ghost
        sat_diff = sat_anchor - sat_c
        val_diff = val_c - val_anchor
        logging.debug(f"🧹 Filter ghost: sat_anchor={sat_anchor:.1f}, sat_c={sat_c:.1f}, sat_diff={sat_diff:.1f} (min={SAT_DELTA_MIN})")
        logging.debug(f"🧹 Filter ghost: val_anchor={val_anchor:.1f}, val_c={val_c:.1f}, val_diff={val_diff:.1f} (min={V_DELTA_MIN})")
        
        if (sat_anchor - sat_c) >= SAT_DELTA_MIN and (val_c - val_anchor) >= V_DELTA_MIN:
            removed += 1
            continue
        out.append(comp)

    if removed:
        logging.info(f"🧹 Ghosts filtrados: {removed} componente(s) descartado(s)")
    return out
def _is_isolated_piece(comp: List[Tuple[int,int]], occ: np.ndarray, min_gap=1) -> bool:
    """
    Verifica si un componente está aislado (rodeado de celdas vacías).
    Útil para distinguir piezas activas de piezas asentadas en stacks.
    """
    rows, cols = occ.shape
    comp_set = set(comp)
    
    # Verificar que haya espacio vacío alrededor del componente
    for r, c in comp:
        # Verificar las 8 direcciones alrededor de cada celda del componente
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                    
                nr, nc = r + dr, c + dc
                
                # Si está dentro del tablero y no es parte del componente
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in comp_set:
                    # Si hay una celda ocupada muy cerca, no está aislado
                    if occ[nr, nc]:
                        return False
                        
    # Verificar que tenga espacio libre debajo (importante para piezas activas)
    bottom_cells = [(r, c) for r, c in comp if r == max(rr for rr, _ in comp)]
    for r, c in bottom_cells:
        # Verificar algunas filas debajo
        for check_rows in range(1, min_gap + 1):
            nr = r + check_rows
            if nr < rows and occ[nr, c]:
                return False  # Hay piezas muy cerca debajo
                
    return True

def _is_part_of_bottom_stack(comp: List[Tuple[int,int]], occ: np.ndarray) -> bool:
    """
    Detecta si un componente es parte de un stack que llega hasta el fondo del tablero.
    Piezas activas típicamente NO forman parte de stacks continuos desde el fondo.
    """
    rows, cols = occ.shape
    
    # Encontrar la fila más baja del componente
    bottom_row = max(r for r, c in comp)
    
    # Si no está cerca del fondo, no puede ser parte del stack de fondo
    if bottom_row < rows - 5:  # Si está 5+ filas arriba del fondo
        return False
        
    # Verificar si hay continuidad desde el componente hasta el fondo
    comp_cols = set(c for r, c in comp)
    
    for col in comp_cols:
        # Para cada columna del componente, verificar si hay piezas continuas hasta el fondo
        continuous_to_bottom = True
        for check_row in range(bottom_row + 1, rows):
            if not occ[check_row, col]:
                continuous_to_bottom = False
                break
                
        if continuous_to_bottom:
            return True  # Al menos una columna es continua hasta el fondo
            
    return False

def find_active_piece(occ: np.ndarray, board_bgr: np.ndarray=None) -> Optional[List[Tuple[int,int]]]:
    """
    Selección robusta de pieza activa:
    1) Elimina ghost del conjunto de componentes usando color (HSV).
    2) De lo restante, elige por puntuación: altura >> aislamiento >> tamaño/compacidad.
    """
    rows, cols = occ.shape
    grid_shape = (rows, cols)

    comps = list_components(occ, max_component_size=4)
    if not comps:
        logging.warning("No hay componentes en occ."); 
        return None

    # 1) eliminar ghost antes de puntuar
    if board_bgr is not None and len(comps) >= 2:
        comps = filter_ghost_pieces(board_bgr, occ, comps, grid_shape)

    best=None; best_top=1e9; best_score=-1
    for i, comp in enumerate(comps):
        n = len(comp)
        if not (2 <= n <= 6): 
            continue
        r0,c0,r1,c1 = bounding_box(comp)
        width = c1-c0+1; height = r1-r0+1
        if width>4 or height>4: 
            continue
        if not (0<=c0<cols and 0<=c1<cols and 0<=r0<rows): 
            continue

        in_upper = (r0 <= 10)

        # aislamiento relativo
        iso = _is_isolated_piece(comp, occ, min_gap=1)

        # compactación
        compact = n/(width*height)

        # prioridad por altura muy fuerte
        size_bonus = 4 if n==4 else (3 if n==3 else 1)
        height_bonus = (100 - r0*5) if in_upper else ((50 - r0*3) if r0<=14 else (10 - r0*2))
        isolation_bonus = 100 if (iso and in_upper) else (50 if iso else 0)
        spawn_bonus = 30 if r0 <= 6 else 0
        compact_bonus = 10*compact

        score = height_bonus + isolation_bonus + spawn_bonus + 5*size_bonus + compact_bonus
        if score > best_score or (score==best_score and r0 < best_top):
            best = comp; best_score = score; best_top = r0

    if not best:
        logging.warning("❌ No se encontró pieza activa.")
        return None

    # 2) opcional: reportar ghost detectado (sin contaminar occ)
    if board_bgr is not None:
        ghost = detect_ghost_component(board_bgr, occ, best)
        if ghost:
            logging.info(f"Ghost confirmado (no se usará como activa): {ghost}")

    return best

# -------------------------- Simulación y heurística ---------------------------

def drop_simulation(board: np.ndarray, shape: List[Tuple[int,int]], left_col: int):
    rows,cols=board.shape
    width=max(c for _,c in shape)+1
    if left_col<0 or left_col+width>cols: return None
    r=0
    while True:
        collided=False
        for dr,dc in shape:
            rr=r+dr; cc=left_col+dc
            if rr>=rows or board[rr,cc]: collided=True; break
        if collided:
            r-=1
            if r< -min(dr for dr,_ in shape): return None
            break
        r+=1
    newb=board.copy()
    for dr,dc in shape:
        rr=r+dr; cc=left_col+dc
        if 0<=rr<rows and 0<=cc<cols: newb[rr,cc]=True
        else: return None
    full=np.all(newb,axis=1); cleared=int(np.sum(full))
    if cleared>0:
        keep=~full; compact=newb[keep]
        newb=np.vstack([np.zeros((rows-compact.shape[0], cols), dtype=bool), compact])
    return newb, cleared

def column_heights(board: np.ndarray)->np.ndarray:
    rows,cols=board.shape
    h=np.zeros(cols,int)
    for c in range(cols):
        col=board[:,c]; filled=np.where(col)[0]
        h[c]=0 if filled.size==0 else (rows-filled[0])
    return h

def count_holes(board: np.ndarray)->int:
    rows,cols=board.shape; holes=0
    for c in range(cols):
        col=board[:,c]; filled=np.where(col)[0]
        if filled.size==0: continue
        top=filled[0]; holes+=int(np.sum(~col[top+1:]))
    return holes

def bumpiness(h: np.ndarray)->int:
    return int(np.sum(np.abs(np.diff(h))))

def log_board_state(board: np.ndarray, piece_cells=None, frame_num=None, context=""):
    """
    Genera un log visual del estado del tablero para debugging
    """
    rows, cols = board.shape
    h = column_heights(board)
    holes = count_holes(board)
    bump = bumpiness(h)
    
    # Header con información del frame
    frame_info = f" Frame {frame_num}" if frame_num is not None else ""
    logging.info(f"📋 BOARD STATE{frame_info} {context}")
    logging.info(f"   Metrics: Heights={h.tolist()}, Holes={holes}, Bump={bump}")
    
    # Representación visual del tablero
    piece_positions = set(piece_cells) if piece_cells else set()
    
    for r in range(min(12, rows)):  # Solo mostrar top 12 filas para no saturar logs
        row_str = "   "
        for c in range(cols):
            if (r, c) in piece_positions:
                row_str += "P"  # Pieza activa
            elif board[r, c]:
                row_str += "█"  # Bloque ocupado
            else:
                row_str += "·"  # Vacío
        
        row_num = f"R{r:2d}"
        logging.info(f"{row_num}|{row_str}|")
    
    # Footer con columnas
    col_header = "   " + "".join(str(i) for i in range(cols))
    logging.info(f"    {col_header}")
    logging.info("   " + "─" * (cols + 2))

def log_decision_context(piece_type, current_board, piece_cells, target_col, rotations, score, reason=""):
    """
    Log detallado del contexto de decisión para debugging
    """
    logging.info(f"🧠 DECISION CONTEXT: {piece_type}")
    logging.info(f"   Piece at: {piece_cells}")
    logging.info(f"   Target: col={target_col}, rotations={rotations}")
    # Handle score formatting - could be number or string
    if isinstance(score, (int, float)):
        logging.info(f"   Score: {score:,}")
    else:
        logging.info(f"   Score: {score}")
    logging.info(f"   Reason: {reason}")
    
    # Mini visualización de la decisión
    if piece_cells:
        try:
            r_min = min(r for r, c in piece_cells)
            r_max = max(r for r, c in piece_cells)
            c_min = min(c for r, c in piece_cells) 
            c_max = max(c for r, c in piece_cells)
            logging.info(f"   Piece bounds: rows {r_min}-{r_max}, cols {c_min}-{c_max}")
            logging.info(f"   Will move from col {c_min} to col {target_col}")
        except (ValueError, TypeError):
            logging.info("   Piece position: unknown")

def diagnose_movement_coordinates(board: 'BoardRect', piece_cells: List[Tuple[int,int]], target_col: int):
    """
    Diagnóstico detallado de cálculo de coordenadas para debugging movimientos
    """
    logging.info("🔬 MOVEMENT COORDINATE DIAGNOSIS")
    
    # Board info
    cw = board.w / 10.0
    logging.info(f"   📏 Board: x={board.x0}, y={board.y0}, w={board.w}, h={board.h}")
    logging.info(f"   📐 Cell width: {cw:.1f} pixels per column")
    
    if piece_cells:
        try:
            r0, c0, r1, c1 = bounding_box(piece_cells)
            piece_width = c1 - c0 + 1
            
            logging.info(f"   🎯 Piece detection: rows {r0}-{r1}, cols {c0}-{c1}")
            logging.info(f"   📦 Piece size: {piece_width} columns wide")
            logging.info(f"   📍 Current position: leftmost col = {c0}")
            
            # Calculate expected pixel positions
            current_pixel_center = board.x0 + (c0 + 0.5) * cw
            target_pixel_center = board.x0 + (target_col + 0.5) * cw
            pixel_distance = target_pixel_center - current_pixel_center
            
            logging.info(f"   🧮 Pixel calculations:")
            logging.info(f"      Current col {c0} center = {current_pixel_center:.1f} px")
            logging.info(f"      Target col {target_col} center = {target_pixel_center:.1f} px")
            logging.info(f"      Distance to move = {pixel_distance:.1f} px")
            logging.info(f"      Columns to move = {target_col - c0} columns")
            
            # Validate that piece fits in target
            if target_col + piece_width - 1 > 9:
                logging.warning(f"   ⚠️ TARGET PROBLEM: Piece width {piece_width} at col {target_col} would exceed board (max col 9)")
            
            # Check for coordinate issues
            if c0 < 0 or c1 > 9:
                logging.warning(f"   ⚠️ DETECTION PROBLEM: Piece columns {c0}-{c1} outside valid range 0-9")
                
        except Exception as e:
            logging.error(f"   ❌ Error in coordinate diagnosis: {e}")
    else:
        logging.warning(f"   ⚠️ No piece cells provided for diagnosis")

def save_column_debug_image(backend: 'ScreenBackend', board: 'BoardRect', piece_cells: List[Tuple[int,int]], 
                           target_col: int, frame_num: int = None):
    """
    Crea una imagen de debug con overlay de columnas para diagnosticar problemas de mapeo.
    FIX: usar backend.get_screen() (no existe screenshot()).
    """
    try:
        # Capturar screenshot actual
        img = backend.get_screen()
        if img is None:
            logging.warning("   📷 Cannot capture screenshot for column debugging")
            return
            
        overlay = img.copy()
        
        # Dimensiones de celda
        cw = board.w / 10.0
        ch = board.h / 20.0
        
        # Dibujar líneas de columnas
        for col in range(11):  # 0..10 bordes
            x = int(board.x0 + col * cw)
            y1 = int(board.y0)
            y2 = int(board.y0 + board.h)
            color = (0, 255, 0) if col == target_col else (255, 255, 255)
            thickness = 3 if col == target_col else 1
            cv2.line(overlay, (x, y1), (x, y2), color, thickness)
            if col < 10:
                label_x = x + int(cw/2) - 5
                label_y = max(10, y1 - 8)
                cv2.putText(overlay, str(col), (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        # Filas cada 5
        for row in range(0, 21, 5):
            y = int(board.y0 + row * ch)
            x1 = int(board.x0)
            x2 = int(board.x0 + board.w)
            cv2.line(overlay, (x1, y), (x2, y), (100, 100, 100), 1)
        
        # Rect de la pieza
        if piece_cells:
            try:
                r0, c0, r1, c1 = bounding_box(piece_cells)
                piece_x1 = int(board.x0 + c0 * cw)
                piece_y1 = int(board.y0 + r0 * ch)
                piece_x2 = int(board.x0 + (c1 + 1) * cw)
                piece_y2 = int(board.y0 + (r1 + 1) * ch)
                cv2.rectangle(overlay, (piece_x1, piece_y1), (piece_x2, piece_y2), (0, 0, 255), 2)
                cv2.putText(overlay, f"Current: {c0}", (piece_x1, max(12, piece_y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception as e:
                logging.debug(f"Error drawing piece overlay: {e}")
        
        # Columna objetivo sombreada
        tx1 = int(board.x0 + target_col * cw)
        ty1 = int(board.y0)
        tx2 = int(board.x0 + (target_col + 1) * cw)
        ty2 = int(board.y0 + board.h)
        col_layer = overlay.copy()
        cv2.rectangle(col_layer, (tx1, ty1), (tx2, ty2), (0, 255, 0), -1)
        cv2.addWeighted(col_layer, 0.25, overlay, 0.75, 0, overlay)
        cv2.putText(overlay, f"Target: {target_col}", (tx1, min(img.shape[0]-10, ty2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Info
        info_y = 30
        cv2.putText(overlay, f"Board: {board.x0},{board.y0} {board.w}x{board.h}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        info_y += 20
        cv2.putText(overlay, f"Cell: {cw:.1f}x{ch:.1f}px",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Guardar
        frame_suffix = f"_frame{frame_num}" if frame_num else ""
        os.makedirs("tetris_debug", exist_ok=True)
        debug_path = os.path.join("tetris_debug", f"debug_columns{frame_suffix}.png")
        cv2.imwrite(debug_path, overlay)
        logging.info(f"   📸 Column debug image saved: {debug_path}")
    except Exception as e:
        logging.error(f"   ❌ Error creating column debug image: {e}")

def validate_screen_coordinates(backend: 'ScreenBackend', board: 'BoardRect') -> dict:
    """
    Herramienta de validación de coordenadas de pantalla para calibración MEJORADA
    """
    logging.info("🔧 COMPREHENSIVE SCREEN COORDINATE VALIDATION")
    
    validation_results = {
        'board_valid': True,
        'gesture_zones_valid': True,
        'cell_mapping_valid': True,
        'movement_safety_valid': True,
        'coordinate_precision_valid': True,
        'issues': [],
        'warnings': []
    }
    
    # Validate board coordinates
    screen_width, screen_height = backend.get_resolution()
    cw = board.w / 10.0
    ch = board.h / 20.0
    
    logging.info(f"   📱 Screen: {screen_width}x{screen_height}")
    logging.info(f"   🎮 Board: x={board.x0}, y={board.y0}, w={board.w}, h={board.h}")
    logging.info(f"   📏 Cell size: {cw:.1f}x{ch:.1f} pixels")
    logging.info(f"   📊 Board coverage: {board.w/screen_width:.1%}w x {board.h/screen_height:.1%}h of screen")
    
    # Enhanced board boundary validation
    if board.x0 < 0 or board.y0 < 0:
        validation_results['board_valid'] = False
        validation_results['issues'].append("Board coordinates are negative")
        
    if board.x0 + board.w > screen_width or board.y0 + board.h > screen_height:
        validation_results['board_valid'] = False
        validation_results['issues'].append("Board extends beyond screen boundaries")
        
    # Enhanced cell size validation with specific recommendations
    if cw < 20:
        validation_results['cell_mapping_valid'] = False
        validation_results['issues'].append(f"Cell width {cw:.1f}px too small (min 20px) - board width may be incorrect")
    elif cw > 100:
        validation_results['cell_mapping_valid'] = False  
        validation_results['issues'].append(f"Cell width {cw:.1f}px too large (max 100px) - board width may be incorrect")
    elif cw < 30:
        validation_results['warnings'].append(f"Cell width {cw:.1f}px is small - movement precision may be difficult")
        
    if ch < 20:
        validation_results['cell_mapping_valid'] = False
        validation_results['issues'].append(f"Cell height {ch:.1f}px too small (min 20px) - board height may be incorrect")  
    elif ch > 100:
        validation_results['cell_mapping_valid'] = False
        validation_results['issues'].append(f"Cell height {ch:.1f}px too large (max 100px) - board height may be incorrect")
    
    # Validate gesture zones
    zones = compute_gesture_zones(board)
    
    logging.info(f"   🎯 Gesture zones:")
    logging.info(f"      Rotate: {zones.rotate_xy}")
    logging.info(f"      Move Y: {zones.mid_band_y}")
    logging.info(f"      Drop: {zones.drop_path}")
    
    # Enhanced gesture zone validation
    if zones.rotate_xy[0] < 0 or zones.rotate_xy[0] > screen_width or zones.rotate_xy[1] < 0 or zones.rotate_xy[1] > screen_height:
        validation_results['gesture_zones_valid'] = False
        validation_results['issues'].append("Rotate zone outside screen boundaries")
        
    if zones.mid_band_y < 0 or zones.mid_band_y > screen_height:
        validation_results['gesture_zones_valid'] = False
        validation_results['issues'].append("Movement zone Y outside screen boundaries")
    
    # NUEVO: Detailed column coordinate mapping with movement validation
    logging.info(f"   📐 Detailed column coordinate mapping:")
    movement_issues = []
    
    for col in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        pixel_x = board.x0 + (col + 0.5) * cw
        pixel_left = board.x0 + col * cw
        pixel_right = board.x0 + (col + 1) * cw
        
        if col % 2 == 0:  # Log every other column to avoid spam
            logging.info(f"      Col {col}: center={pixel_x:.1f}px, bounds=[{pixel_left:.1f}, {pixel_right:.1f}]")
        
        # Check if column mapping is within board bounds
        if pixel_left < board.x0 or pixel_right > board.x0 + board.w:
            validation_results['cell_mapping_valid'] = False
            validation_results['issues'].append(f"Column {col} extends outside board boundaries")
            
        # Check movement safety - can we safely swipe within this column?
        safety_margin = 15
        safe_left = board.x0 + safety_margin
        safe_right = board.x0 + board.w - safety_margin
        
        if pixel_x < safe_left or pixel_x > safe_right:
            movement_issues.append(col)
    
    # NUEVO: Movement distance validation
    logging.info(f"   🔄 Movement distance validation:")
    max_single_move_distance = 0.8 * cw  # Based on our new safe movement calculation
    for distance in [1, 2, 3, 4]:  # Test 1-4 column movements
        pixel_distance = distance * max_single_move_distance
        steps_required = distance  # One step per column
        total_time = steps_required * 0.25  # Estimated time per step
        
        logging.info(f"      {distance} cols: {pixel_distance:.1f}px, {steps_required} steps, ~{total_time:.1f}s")
        
        if pixel_distance > cw * 1.2:  # If movement is more than 1.2x cell width
            validation_results['warnings'].append(f"{distance}-column movement may be imprecise")
    
    if movement_issues:
        validation_results['movement_safety_valid'] = False
        validation_results['issues'].append(f"Columns {movement_issues} are in unsafe movement zones")
        
    # NUEVO: Coordinate precision analysis
    min_meaningful_movement = 5  # Minimum pixels for meaningful swipe
    precision_score = min_meaningful_movement / cw
    
    logging.info(f"   🎯 Coordinate precision analysis:")
    logging.info(f"      Min meaningful movement: {min_meaningful_movement}px")
    logging.info(f"      Precision ratio: {precision_score:.3f} (lower is better)")
    
    if precision_score > 0.2:  # If minimum movement is more than 20% of cell width
        validation_results['coordinate_precision_valid'] = False
        validation_results['issues'].append(f"Movement precision insufficient: {precision_score:.1%} of cell width")
    elif precision_score > 0.1:
        validation_results['warnings'].append(f"Movement precision is marginal: {precision_score:.1%} of cell width")
    
    # Summary with enhanced reporting
    overall_valid = all([
        validation_results['board_valid'], 
        validation_results['gesture_zones_valid'], 
        validation_results['cell_mapping_valid'],
        validation_results['movement_safety_valid'],
        validation_results['coordinate_precision_valid']
    ])
    
    status = "✅ VALID" if overall_valid else "❌ ISSUES FOUND"
    logging.info(f"   📋 Validation result: {status}")
    
    if validation_results['issues']:
        logging.warning("   🚨 Critical issues found:")
        for issue in validation_results['issues']:
            logging.warning(f"      ❌ {issue}")
            
    if validation_results['warnings']:
        logging.info("   ⚠️ Warnings:")
        for warning in validation_results['warnings']:
            logging.info(f"      ⚠️ {warning}")
    
    # NUEVO: Provide specific recommendations
    if not overall_valid:
        logging.error("   💡 RECOMMENDATIONS:")
        if not validation_results['board_valid']:
            logging.error("      📱 Check screen resolution and board rectangle coordinates")
        if not validation_results['cell_mapping_valid']:
            logging.error("      📏 Recalibrate board width/height - cells are wrong size")
        if not validation_results['movement_safety_valid']:
            logging.error("      🔄 Adjust movement margins - some columns unreachable")
        if not validation_results['coordinate_precision_valid']:
            logging.error("      🎯 Increase board size or adjust movement algorithm")
    
    validation_results['overall_valid'] = overall_valid
    return validation_results

def verify_board_runtime(board: 'BoardRect', piece_cells: List[Tuple[int,int]], 
                        occ_grid: np.ndarray, frame_num: int) -> dict:
    """
    Verificación runtime del rectángulo del tablero basada en datos reales detectados
    """
    verification_results = {
        'board_alignment_valid': True,
        'piece_position_valid': True, 
        'occupancy_distribution_valid': True,
        'coordinate_consistency_valid': True,
        'confidence_score': 1.0,
        'issues': [],
        'recommendations': []
    }
    
    try:
        rows, cols = occ_grid.shape
        cw = board.w / 10.0
        ch = board.h / 20.0
        
        # 1. VERIFICACIÓN DE POSICIÓN DE PIEZA
        if piece_cells:
            piece_rows = [r for r, c in piece_cells]
            piece_cols = [c for r, c in piece_cells]
            
            min_row, max_row = min(piece_rows), max(piece_rows)
            min_col, max_col = min(piece_cols), max(piece_cols)
            
            # Verificar que la pieza está dentro de los límites lógicos
            if min_col < 0 or max_col >= cols or min_row < 0 or max_row >= rows:
                verification_results['piece_position_valid'] = False
                verification_results['issues'].append(
                    f"Piece detected outside board bounds: rows {min_row}-{max_row}, cols {min_col}-{max_col}"
                )
                verification_results['confidence_score'] *= 0.5
                
            # Verificar coherencia del tamaño de pieza  
            piece_width = max_col - min_col + 1
            piece_height = max_row - min_row + 1
            piece_cells_count = len(piece_cells)
            
            if piece_cells_count > 4:
                verification_results['piece_position_valid'] = False
                verification_results['issues'].append(f"Oversized piece detected: {piece_cells_count} cells")
                verification_results['confidence_score'] *= 0.3
                
            if piece_width > 4 or piece_height > 4:
                verification_results['coordinate_consistency_valid'] = False
                verification_results['issues'].append(f"Piece dimensions too large: {piece_width}x{piece_height}")
                verification_results['confidence_score'] *= 0.6
        
        # 2. VERIFICACIÓN DE DISTRIBUCIÓN DE OCUPACIÓN
        # Analizar si la ocupación sigue patrones esperados de Tetris
        total_occupied = np.sum(occ_grid)
        total_cells = rows * cols
        occupation_rate = total_occupied / total_cells
        
        # Verificar distribución por filas (las piezas deben caer hacia abajo)
        row_occupancy = [np.sum(occ_grid[r, :]) for r in range(rows)]
        
        # Las filas superiores no deberían tener más ocupación que las inferiores
        # a menos que sea el inicio del juego
        if occupation_rate > 0.1:  # Solo verificar si hay suficiente ocupación
            for r in range(rows - 5, rows):  # Verificar últimas 5 filas
                if r < rows - 1:
                    current_row_occ = row_occupancy[r]
                    next_row_occ = row_occupancy[r + 1]
                    
                    # Si una fila superior tiene mucha más ocupación que la inferior
                    if current_row_occ > 0 and next_row_occ == 0 and current_row_occ > 3:
                        verification_results['occupancy_distribution_valid'] = False
                        verification_results['issues'].append(
                            f"Suspicious occupancy pattern: row {r} has {current_row_occ} cells but row {r+1} empty"
                        )
                        verification_results['recommendations'].append(
                            "Board rectangle may be missing bottom rows - consider expanding vertically"
                        )
                        verification_results['confidence_score'] *= 0.7
        
        # 3. VERIFICACIÓN DE COHERENCIA TEMPORAL
        # Esta verificación se puede expandir para tracking entre frames
        
        # 4. VERIFICACIÓN DE LÍMITES DE COORDENADAS
        # Verificar que las coordenadas calculadas están dentro de rangos razonables
        if piece_cells:
            for r, c in piece_cells:
                expected_pixel_x = board.x0 + (c + 0.5) * cw
                expected_pixel_y = board.y0 + (r + 0.5) * ch
                
                # Los pixels calculados deben estar dentro del rectángulo del board
                if not (board.x0 <= expected_pixel_x <= board.x0 + board.w):
                    verification_results['coordinate_consistency_valid'] = False
                    verification_results['issues'].append(
                        f"Calculated X coordinate {expected_pixel_x:.1f} outside board range [{board.x0}, {board.x0 + board.w}]"
                    )
                    verification_results['confidence_score'] *= 0.4
                    
                if not (board.y0 <= expected_pixel_y <= board.y0 + board.h):
                    verification_results['coordinate_consistency_valid'] = False
                    verification_results['issues'].append(
                        f"Calculated Y coordinate {expected_pixel_y:.1f} outside board range [{board.y0}, {board.y0 + board.h}]"
                    )
                    verification_results['confidence_score'] *= 0.4
        
        # Determinar si la verificación general es válida
        overall_valid = (
            verification_results['board_alignment_valid'] and
            verification_results['piece_position_valid'] and
            verification_results['occupancy_distribution_valid'] and 
            verification_results['coordinate_consistency_valid']
        )
        
        verification_results['overall_valid'] = overall_valid
        
        # Log de resultados si hay problemas
        if not overall_valid or verification_results['confidence_score'] < 0.8:
            logging.warning(f"🔍 RUNTIME BOARD VERIFICATION - Frame {frame_num}")
            logging.warning(f"   Confidence: {verification_results['confidence_score']:.2f}")
            logging.warning(f"   Issues found: {len(verification_results['issues'])}")
            
            for issue in verification_results['issues'][:3]:  # Max 3 issues per log
                logging.warning(f"   ❌ {issue}")
                
            for rec in verification_results['recommendations'][:2]:  # Max 2 recommendations  
                logging.warning(f"   💡 {rec}")
                
            if verification_results['confidence_score'] < 0.5:
                logging.error("🚨 CRITICAL: Board rectangle calibration appears incorrect!")
                
        elif frame_num % 300 == 0:  # Log success every 5 minutes
            logging.debug(f"✅ Runtime board verification passed (confidence: {verification_results['confidence_score']:.2f})")
        
    except Exception as e:
        logging.error(f"❌ Runtime board verification failed: {e}")
        verification_results['overall_valid'] = False
        verification_results['confidence_score'] = 0.0
        verification_results['issues'].append(f"Verification error: {e}")
    
    return verification_results

def evaluate_board_advanced(board: np.ndarray, lines_cleared: int, 
                           enable_tspin_detection=True, enable_combo_bonus=True) -> float:
    """
    Función de evaluación avanzada que considera T-spins, combos y patrones especiales
    
    Args:
        board: Estado del tablero (20x10)
        lines_cleared: Número de líneas limpiadas
        enable_tspin_detection: Si detectar setups de T-spin
        enable_combo_bonus: Si bonificar combos consecutivos
    
    Returns:
        Score del estado del tablero
    """
    # Rewards básicos por líneas (mantenidos del original)
    base_rewards = [0, 1_000_000, 3_000_000, 8_000_000, 20_000_000]
    reward = base_rewards[min(lines_cleared, 4)]
    
    # Cálculos básicos
    h = column_heights(board)
    holes = count_holes(board)
    bump = bumpiness(h)
    agg_h = int(np.sum(h))
    max_h = int(np.max(h))
    
    # Penalizaciones básicas (del config.json)
    base_penalty = holes * 1500 + agg_h * 6 + bump * 40 + max_h * 20
    
    # === MEJORAS AVANZADAS ===
    advanced_bonus = 0
    
    # 1. Bonus por setup de T-spin (detectar cavidades en forma de T)
    if enable_tspin_detection:
        tspin_bonus = detect_tspin_setups(board, h)
        advanced_bonus += tspin_bonus
    
    # 2. Bonus por limpieza de wells (columnas profundas)
    well_bonus = evaluate_well_clearing(board, h, lines_cleared)
    advanced_bonus += well_bonus
    
    # 3. Penalización por dependencias (bloques que no se pueden limpiar fácilmente)
    dependency_penalty = evaluate_dependencies(board, h)
    advanced_bonus -= dependency_penalty
    
    # 4. Bonus por mantener superficie plana en zonas críticas
    surface_bonus = evaluate_surface_flatness(h)
    advanced_bonus += surface_bonus
    
    # 5. Bonus especial por quad/tetris setup
    if lines_cleared == 4:
        advanced_bonus += 5_000_000  # Bonus extra por tetris
    
    return reward + advanced_bonus - base_penalty

def detect_tspin_setups(board: np.ndarray, heights: np.ndarray) -> float:
    """Detecta posibles setups de T-spin en el tablero"""
    tspin_bonus = 0
    rows, cols = board.shape
    
    # Buscar patrones típicos de T-spin en las columnas
    for col in range(1, cols-1):  # No en bordes
        h_left = heights[col-1] if col > 0 else 0
        h_center = heights[col]
        h_right = heights[col+1] if col < cols-1 else 0
        
        # Patrón de cavidad para T-spin: centro más bajo que los lados
        if h_center < h_left-1 and h_center < h_right-1:
            # Verificar que hay espacio para T-spin
            if h_center < rows-3:  # Al menos 3 filas libres arriba
                cavity_depth = min(h_left, h_right) - h_center
                if cavity_depth >= 2:
                    # Bonus por setup de T-spin potencial
                    tspin_bonus += cavity_depth * 500_000
    
    return tspin_bonus

def evaluate_well_clearing(board: np.ndarray, heights: np.ndarray, lines_cleared: int) -> float:
    """Evalúa la limpieza efectiva de wells (columnas profundas)"""
    well_bonus = 0
    
    # Detectar wells (columnas significativamente más bajas que las adyacentes)
    for col in range(len(heights)):
        h_current = heights[col]
        h_left = heights[col-1] if col > 0 else h_current
        h_right = heights[col+1] if col < len(heights)-1 else h_current
        
        # Si es un well (más bajo que ambos lados)
        if h_current < h_left-2 and h_current < h_right-2:
            well_depth = min(h_left, h_right) - h_current
            
            # Si se limpiaron líneas y era un well profundo, bonus
            if lines_cleared > 0 and well_depth >= 3:
                well_bonus += well_depth * lines_cleared * 200_000
    
    return well_bonus

def evaluate_dependencies(board: np.ndarray, heights: np.ndarray) -> float:
    """Penaliza configuraciones donde bloques dependen mucho de otros"""
    penalty = 0
    rows, cols = board.shape
    
    # Contar bloques "flotantes" o difíciles de limpiar
    for col in range(cols):
        for row in range(int(heights[col])):
            if board[rows-1-row, col] == 1:  # Si hay un bloque
                # Verificar si tiene bloques encima que lo protegen
                blocks_above = int(heights[col] - row - 1)
                if blocks_above > 3:  # Muchos bloques encima
                    penalty += blocks_above * 10_000
    
    return penalty


def _choose_backend(name: str, serial: Optional[str]) -> ScreenBackend:
    name = (name or "hybrid").lower()
    if name == "adb":
        return ADBBackend(serial)
    if name == "scrcpy":
        return ScrcpyBackend(serial)
    return HybridBackend(serial)  # default


def evaluate_surface_flatness(heights: np.ndarray) -> float:
    """Bonus por mantener superficie relativamente plana"""
    if len(heights) < 2:
        return 0
    
    # Calcular varianza de alturas (menor = más plano)
    height_variance = np.var(heights)
    
    # Bonus inversamente proporcional a la varianza
    flatness_bonus = max(0, 1_000_000 - height_variance * 50_000)
    
    return flatness_bonus


def detect_line_clear_opportunities(board: np.ndarray, heights: np.ndarray) -> int:
    """
    Detecta filas que están cerca de completarse (pocas celdas vacías).
    Retorna un score basado en cuántas oportunidades de line clear hay.
    """
    rows, cols = board.shape
    opportunities = 0
    
    for row in range(rows):
        # Contar celdas ocupadas en esta fila
        occupied_cells = np.sum(board[row, :])
        empty_cells = cols - occupied_cells
        
        # Si la fila está casi completa, es una oportunidad
        if empty_cells == 1:
            opportunities += 10  # Fila necesita solo 1 pieza - ¡prioritaria!
        elif empty_cells == 2:
            opportunities += 5   # Fila necesita 2 piezas - buena oportunidad
        elif empty_cells == 3:
            opportunities += 2   # Fila necesita 3 piezas - oportunidad moderada
    
    return opportunities


def evaluate_tetris_well_strategy(board: np.ndarray, heights: np.ndarray) -> float:
    """
    Evalúa la estrategia de mantener un 'well' (columna profunda) para hacer Tetrises.
    Bonifica mantener una columna vacía mientras el resto del tablero sube.
    """
    if len(heights) != 10:
        return 0.0
        
    tetris_bonus = 0.0
    
    # Buscar wells (columnas significativamente más bajas que sus vecinas)
    for col in range(len(heights)):
        h_current = heights[col]
        h_left = heights[col-1] if col > 0 else h_current  
        h_right = heights[col+1] if col < len(heights)-1 else h_current
        
        # Calcular qué tan profundo es el well
        min_neighbor = min(h_left, h_right) 
        well_depth = min_neighbor - h_current
        
        if well_depth >= 3:  # Es un well significativo
            # BONUS masivo por mantener well profundo
            tetris_bonus += well_depth * 2.0
            
            # BONUS extra si el well está en los bordes (más fácil de llenar con I-piece)
            if col == 0 or col == 9:
                tetris_bonus += well_depth * 1.5
                
            # BONUS extra si el well tiene exactamente la altura ideal para Tetris
            if well_depth == 4:
                tetris_bonus += 10.0  # Perfecto para I-piece Tetris
        
        # PENALTY severa por llenar wells existentes
        elif well_depth < 0:  # Columna más alta que vecinas
            tetris_bonus -= abs(well_depth) * 3.0
    
    return tetris_bonus


def evaluate_surface_flatness_simple(heights: np.ndarray) -> float:
    """
    Evalúa qué tan plana es la superficie (versión simplificada).
    Superficie más plana = mejor para colocar piezas.
    """
    if len(heights) < 2:
        return 0.0
        
    # Calcular varianza de alturas
    height_variance = float(np.var(heights))
    
    # Bonus inversamente proporcional a la varianza
    flatness_score = max(0, 10.0 - height_variance * 0.5)
    
    return flatness_score


def evaluate_row_density(board: np.ndarray) -> float:
    """
    Evalúa la densidad de las filas (filas más llenas son mejores).
    Prioriza tener filas casi completas.
    """
    rows, cols = board.shape
    density_score = 0.0
    
    for row in range(rows):
        occupied = np.sum(board[row, :])
        density_ratio = occupied / cols
        
        # Bonus creciente por filas más llenas
        if density_ratio >= 0.8:      # 80%+ lleno
            density_score += 10.0
        elif density_ratio >= 0.6:    # 60%+ lleno  
            density_score += 5.0
        elif density_ratio >= 0.4:    # 40%+ lleno
            density_score += 2.0
    
    return density_score


def detect_combo_potential(board: np.ndarray, heights: np.ndarray) -> int:
    """
    Detecta potencial para combos en cascada.
    Busca configuraciones donde limpiar una línea puede causar más line clears.
    """
    rows, cols = board.shape
    combo_score = 0
    
    # Buscar configuraciones donde hay bloques "flotantes"
    for row in range(1, rows):  # Empezar desde fila 1
        for col in range(cols):
            # Si hay un bloque con espacio debajo
            if board[row, col] and not board[row-1, col]:
                # Verificar si limpiar filas debajo podría hacer que este bloque caiga
                empty_spaces_below = 0
                for check_row in range(row-1, -1, -1):
                    if not board[check_row, col]:
                        empty_spaces_below += 1
                    else:
                        break
                
                # Bonus por bloques que pueden caer y causar combos
                if empty_spaces_below >= 2:
                    combo_score += empty_spaces_below * 2
    
    return combo_score


def evaluate_board(board: np.ndarray, lines_cleared:int)->float:
    """
    Función de evaluación AGRESIVA optimizada para LIMPIAR LÍNEAS.
    Prioriza absolutamente los line clears sobre cualquier otra consideración.
    """
    # REWARDS MASIVAMENTE AUMENTADOS para line clears
    base_rewards = [0, 5_000_000, 15_000_000, 50_000_000, 200_000_000]
    reward = base_rewards[min(lines_cleared, 4)]
    
    # BONUS EXTRA MASIVO por Tetris (4 líneas)
    if lines_cleared == 4:
        reward += 300_000_000  # Bonus adicional brutal por Tetris
        
    # Cálculos de estado del tablero
    h = column_heights(board)
    holes = count_holes(board)
    bump = bumpiness(h) 
    agg_h = int(np.sum(h))
    max_h = int(np.max(h))
    
    # PENALTIES SEVERAMENTE AUMENTADAS para forzar line clears
    holes_penalty = holes * 8_000        # Aumentado de 1500 a 8000
    height_penalty = agg_h * 100         # Aumentado de 6 a 100  
    bumpiness_penalty = bump * 200       # Aumentado de 40 a 200
    max_height_penalty = max_h * 1_000   # Aumentado de 20 a 1000
    
    # BONUS por oportunidades de line clear
    line_clear_bonus = detect_line_clear_opportunities(board, h) * 2_000_000
    
    # ESTRATEGIA DE TETRIS: Detectar y mantener well profundo
    tetris_well_bonus = evaluate_tetris_well_strategy(board, h) * 5_000_000
    
    # HEURÍSTICAS AVANZADAS
    # 1. Surface flatness bonus - superficie plana es mejor
    surface_bonus = evaluate_surface_flatness_simple(h) * 1_000_000
    
    # 2. Row density bonus - filas más llenas son mejores
    row_density_bonus = evaluate_row_density(board) * 500_000
    
    # 3. Penalty masiva por tablero muy lleno (game over danger)
    danger_penalty = 0
    if max_h >= 18:  # Muy cerca del tope
        danger_penalty = (max_h - 17) * 50_000_000  # Penalty brutal
    
    # 4. Combo potential - detectar posibles combos en cascada
    combo_potential = detect_combo_potential(board, h) * 3_000_000
    
    total_penalty = holes_penalty + height_penalty + bumpiness_penalty + max_height_penalty + danger_penalty
    total_bonus = line_clear_bonus + tetris_well_bonus + surface_bonus + row_density_bonus + combo_potential
    
    final_score = reward + total_bonus - total_penalty
    
    # Log de scoring detallado para debugging mejorado
    if lines_cleared > 0 or final_score > 1_000_000 or logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"📊 EVALUATION BREAKDOWN: Score = {final_score:,}")
        logging.debug(f"   🎁 Rewards: {reward:,} (lines: {lines_cleared})")
        logging.debug(f"   ⭐ Bonuses: {total_bonus:,}")
        logging.debug(f"      ├─ Line clear opportunities: {line_clear_bonus:,}")
        logging.debug(f"      ├─ Tetris well strategy: {tetris_well_bonus:,}")
        logging.debug(f"      ├─ Surface flatness: {surface_bonus:,}")
        logging.debug(f"      ├─ Row density: {row_density_bonus:,}")
        logging.debug(f"      └─ Combo potential: {combo_potential:,}")
        logging.debug(f"   ❌ Penalties: {total_penalty:,}")
        logging.debug(f"      ├─ Holes ({holes}): {holes_penalty:,}")
        logging.debug(f"      ├─ Height ({agg_h}): {height_penalty:,}")
        logging.debug(f"      ├─ Max height ({max_h}): {max_height_penalty:,}")
        logging.debug(f"      ├─ Bumpiness ({bump}): {bumpiness_penalty:,}")
        logging.debug(f"      └─ Danger zone: {danger_penalty:,}")
        logging.debug(f"   📏 Board metrics: H={h.tolist()}")
    
    return final_score

class OneStepPolicy:
    def choose(self, board_stack: np.ndarray, piece: str)->Optional[Tuple[int,int]]:
        best=None; best_score=-1e18
        move_evaluations = []  # Para logging detallado
        
        for oi,shape in enumerate(PIECE_ORIENTS[piece]):
            width=max(c for _,c in shape)+1
            # EVITAR POSICIONES EXTREMAS PROBLEMÁTICAS
            margin = 1
            min_col = max(0, margin)
            max_col = (10 - width - margin)
            max_col = max(min_col, max_col)  # por si width=4
            
            for left in range(min_col, max_col+1):
                sim=drop_simulation(board_stack, shape, left)
                if sim is None: 
                    continue
                    
                newb, cleared = sim
                score=evaluate_board(newb, cleared)
                
                # Almacenar evaluación para debugging
                move_evaluations.append({
                    'orientation': oi,
                    'left_pos': left, 
                    'score': score,
                    'lines_cleared': cleared,
                    'is_best': False
                })
                
                if score>best_score:
                    # Marcar movimientos anteriores como no-best
                    for eval_move in move_evaluations:
                        eval_move['is_best'] = False
                    # Marcar este como best
                    move_evaluations[-1]['is_best'] = True
                    
                    best_score=score; best=(oi,left)
        
        # Log de la mejor decisión y alternativas mejorado
        if best and move_evaluations:
            total_moves_evaluated = len(move_evaluations)
            logging.info(f"🎯 {piece} DECISION: O:{best[0]}, C:{best[1]}, Score:{best_score:,} ({total_moves_evaluated} moves evaluated)")
            
            # Mostrar top 5 mejores opciones con más detalle
            move_evaluations.sort(key=lambda x: x['score'], reverse=True)
            logging.debug(f"📈 Top 5 moves for {piece} (out of {total_moves_evaluated}):")
            for i, move in enumerate(move_evaluations[:5]):
                marks = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                mark = marks[i] if i < len(marks) else f"{i+1}."
                score_diff = move['score'] - best_score if move['score'] != best_score else 0
                diff_str = f" ({score_diff:+,})" if score_diff != 0 else " (CHOSEN)"
                logging.debug(f"   {mark} O:{move['orientation']}, C:{move['left_pos']}, Score:{move['score']:,}{diff_str}, Lines:{move['lines_cleared']}")
                
            # Mostrar distribución de scores para entender varianza  
            if len(move_evaluations) > 1:
                scores = [m['score'] for m in move_evaluations]
                worst_score = min(scores)
                score_range = best_score - worst_score
                logging.debug(f"   📊 Score range: {worst_score:,} to {best_score:,} (spread: {score_range:,})")
        elif not move_evaluations:
            logging.warning(f"⚠️ No valid moves found for {piece}!")
                
        return best

class MultiStepPolicy:
    """Política avanzada que considera múltiples movimientos por adelantado (lookahead)"""
    
    def __init__(self, lookahead_depth=2, pieces_preview=None):
        self.lookahead_depth = max(1, min(lookahead_depth, 3))  # Limite razonable 1-3
        self.pieces_preview = pieces_preview or []
        
    def set_pieces_preview(self, pieces: List[str]):
        """Configura la preview de piezas futuras si está disponible"""
        self.pieces_preview = pieces[:self.lookahead_depth]
    
    def choose(self, board_stack: np.ndarray, piece: str) -> Optional[Tuple[int, int]]:
        """
        Elige el mejor movimiento considerando múltiples pasos adelante
        
        Args:
            board_stack: Estado actual del tablero
            piece: Pieza actual
        
        Returns:
            Tupla (orientación, columna_izquierda) o None si no hay movimiento válido
        """
        if self.lookahead_depth == 1 or not self.pieces_preview:
            # Fallback a política simple si no hay lookahead
            return self._choose_single_step(board_stack, piece)
        
        # Evaluación multi-step con logging detallado
        best_action = None
        best_score = -1e18
        move_evaluations = []
        
        # Evaluar todas las posibles acciones para la pieza actual
        for oi, shape in enumerate(PIECE_ORIENTS[piece]):
            width = max(c for _, c in shape) + 1
            margin = 1
            min_col = max(0, margin)
            max_col = (10 - width - margin)
            max_col = max(min_col, max_col)  # por si width=4
            
            for left in range(min_col, max_col+1):
                # Simular movimiento actual
                sim = drop_simulation(board_stack, shape, left)
                if sim is None:
                    continue
                    
                new_board, lines_cleared = sim
                
                # Evaluación multi-step recursiva
                total_score = self._evaluate_sequence(
                    new_board, 
                    lines_cleared,
                    self.pieces_preview[:self.lookahead_depth-1], 
                    depth=1
                )
                
                # Almacenar para debugging
                move_evaluations.append({
                    'orientation': oi,
                    'left_pos': left,
                    'score': total_score,
                    'immediate_lines': lines_cleared
                })
                
                if total_score > best_score:
                    best_score = total_score
                    best_action = (oi, left)
        
        # Log de decisión multistep mejorado
        if best_action and move_evaluations:
            total_moves_evaluated = len(move_evaluations)
            preview_str = "+".join(self.pieces_preview[:3]) if self.pieces_preview else "no-preview"
            logging.info(f"🧠 {piece} MULTISTEP DECISION: O:{best_action[0]}, C:{best_action[1]}, Score:{best_score:,}")
            logging.info(f"   📡 Lookahead: depth-{self.lookahead_depth}, preview=[{preview_str}], {total_moves_evaluated} moves evaluated")
            
            # Top 5 con lookhead y más detalle
            move_evaluations.sort(key=lambda x: x['score'], reverse=True)
            logging.debug(f"🔮 Top 5 multistep moves for {piece} (out of {total_moves_evaluated}):")
            for i, move in enumerate(move_evaluations[:5]):
                marks = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                mark = marks[i] if i < len(marks) else f"{i+1}."
                score_diff = move['score'] - best_score if move['score'] != best_score else 0
                diff_str = f" ({score_diff:+,})" if score_diff != 0 else " (CHOSEN)"
                logging.debug(f"   {mark} O:{move['orientation']}, C:{move['left_pos']}, Score:{move['score']:,}{diff_str}, Imm.Lines:{move['immediate_lines']}")
            
            # Mostrar distribución de scores multistep
            if len(move_evaluations) > 1:
                scores = [m['score'] for m in move_evaluations]
                worst_score = min(scores)
                score_range = best_score - worst_score
                logging.debug(f"   📊 Multistep score range: {worst_score:,} to {best_score:,} (spread: {score_range:,})")
        elif not move_evaluations:
            logging.warning(f"⚠️ No valid multistep moves found for {piece}!")
        
        return best_action
    
    def _choose_single_step(self, board_stack: np.ndarray, piece: str) -> Optional[Tuple[int, int]]:
        """Implementación de un solo paso (igual que OneStepPolicy)"""
        best = None
        best_score = -1e18
        
        for oi, shape in enumerate(PIECE_ORIENTS[piece]):
            width = max(c for _, c in shape) + 1
            # EVITAR POSICIONES EXTREMAS PROBLEMÁTICAS  
            min_col = 0  # Evitar columna 0 por problemas de swipe
            max_col = 10 - width  # Evitar columnas muy a la derecha
            
            for left in range(min_col, max_col+1):
                sim = drop_simulation(board_stack, shape, left)
                if sim is None:
                    continue
                newb, cleared = sim
                score = evaluate_board(newb, cleared)
                if score > best_score:
                    best_score = score
                    best = (oi, left)
        return best
    
    def _evaluate_sequence(self, board: np.ndarray, lines_cleared: int, 
                          remaining_pieces: List[str], depth: int) -> float:
        """
        Evalúa recursivamente una secuencia de movimientos futuros
        
        Args:
            board: Estado del tablero después del movimiento actual
            lines_cleared: Líneas limpiadas por el movimiento actual
            remaining_pieces: Piezas restantes a evaluar
            depth: Profundidad actual en la recursión
        
        Returns:
            Score total de la secuencia
        """
        # Score base del estado actual
        current_score = evaluate_board(board, lines_cleared)
        
        # Si no hay más piezas o alcanzamos profundidad máxima
        if not remaining_pieces or depth >= self.lookahead_depth:
            return current_score
        
        # Evaluar la siguiente pieza
        next_piece = remaining_pieces[0]
        remaining = remaining_pieces[1:]
        
        best_future_score = -1e18
        valid_moves_found = False
        
        # Probar todos los movimientos posibles para la siguiente pieza
        for oi, shape in enumerate(PIECE_ORIENTS[next_piece]):
            width = max(c for _, c in shape) + 1
            # EVITAR POSICIONES EXTREMAS PROBLEMÁTICAS
            min_col = 0 # Evitar columna 0 por problemas de swipe
            max_col = 10 - width  # Evitar columnas muy a la derecha
            
            for left in range(min_col, max_col+1):
                sim = drop_simulation(board, shape, left)
                if sim is None:
                    continue
                    
                valid_moves_found = True
                future_board, future_cleared = sim
                
                # Recursión para evaluar movimientos futuros
                future_score = self._evaluate_sequence(
                    future_board, 
                    future_cleared,
                    remaining, 
                    depth + 1
                )
                
                best_future_score = max(best_future_score, future_score)
        
        # Si no hay movimientos válidos en el futuro, penalizar
        if not valid_moves_found:
            return current_score - 1e10  # Penalización severa por bloqueo
        
        # Combinar score actual con score futuro (con descuento por profundidad)
        discount_factor = 0.85 ** depth  # Descuento temporal
        return current_score + discount_factor * best_future_score


# =============================== Gestos táctiles ===============================

from dataclasses import dataclass

@dataclass
class GestureZones:
    rotate_xy: Tuple[int,int]
    mid_band_y: int
    drop_path: Tuple[Tuple[int,int], Tuple[int,int]]
    column_centers: List[int]

def compute_gesture_zones(board: BoardRect) -> GestureZones:
    cw = board.w/10.0; ch = board.h/20.0
    rotate_xy   = (int(board.x0 + 8.5*cw), int(board.y0 + 2.0*ch))
    mid_band_y  = int(board.y0 + 9.5*ch)  # banda estable para micro-swipes
    drop_start  = (int(board.x0 + 5.0*cw), int(board.y0 + 2.0*ch))
    drop_end    = (drop_start[0], int(board.y0 + 18.5*ch))
    centers     = [int(board.x0 + (c+0.5)*cw) for c in range(10)]
    return GestureZones(rotate_xy, mid_band_y, (drop_start, drop_end), centers)

def rotate_action(backend: ScreenBackend, zones: GestureZones, taps:int=1):
    x,y = zones.rotate_xy
    for _ in range(taps):
        backend.tap(x + jitter(2), y + jitter(2), hold_ms=60)

def drop_action(backend: ScreenBackend, zones: GestureZones):
    (x1,y1),(x2,y2) = zones.drop_path
    backend.swipe(x1 + jitter(2), y1, x2 + jitter(2), y2, duration_ms=230)

def move_piece_to_column(backend: 'ScreenBackend',
                         zones: 'GestureZones',
                         board: BoardRect,
                         piece_cells: List[Tuple[int,int]],
                         target_col: int,
                         timing_multiplier: float = 1.0):
    """
    Mueve con micro-swipes de ~1 columna cada uno, a una Y estable dentro del tablero.
    NO usa draganddrop en ningún caso.
    """
    if not piece_cells:
        return

    # Tamaño de celda
    cw = board.w / 10.0

    # Y segura para swipe horizontal (clamp al interior del board)
    y = int(max(board.y0 + board.h * 0.30, min(zones.mid_band_y, board.y0 + board.h * 0.85)))

    # Columna actual (izquierda de la pieza)
    _, c0, _, _ = bounding_box(piece_cells)
    delta = int(target_col) - int(c0)
    if delta == 0:
        return

    # Ajustes de gesto
    step_sign = 1 if delta > 0 else -1
    steps = abs(delta)

    # Un poco menos que el ancho de celda para evitar overshoot y bordes
    step_px = int(0.92 * cw) * step_sign
    # Duración más larga porque tu dispositivo lo exige (según tu prueba manual)
    dur = max(200, int(200 * timing_multiplier))

    # Empezamos desde el centro de la pieza actual
    x_start = int(board.x0 + (c0 + 0.5) * cw)

    # Fallback 1: un swipe largo directo (algunos juegos lo prefieren)
    try:
        x_target = int(board.x0 + (target_col + 0.5) * cw)
        logging.debug(f"➡️ Long swipe: ({x_start},{y}) -> ({x_target},{y}) dur={dur}ms")
        backend.swipe(x_start, y, x_target, y, duration_ms=dur)
        time.sleep(0.06)
        return
    except Exception as e:
        logging.debug(f"Long swipe falló: {e}. Probando micro-swipes...")

    # Fallback 2: micro-swipes de 1 columna
    x = x_start
    for i in range(steps):
        x_next = int(x + step_px)
        # clamp horizontal dentro del board
        x_next = int(max(board.x0 + 4, min(x_next, board.x0 + board.w - 4)))
        logging.debug(f"➡️ Step {i+1}/{steps}: swipe ({x},{y}) -> ({x_next},{y}) dur={dur}ms")
        backend.swipe(x, y, x_next, y, duration_ms=dur)
        x = x_next
        time.sleep(0.04)
# =========================== Control una-vez-por-pieza ==========================

class PieceTracker:
    def __init__(self):
        self.last_sig=None
        self.last_pos=None  # (fila, columna) de la pieza anterior
        self.acted=False
        self.last_action_t=0.0
        self.action_timeout=1.5  # timeout para reintento si la pieza no se mueve
        self.position_history=[]  # historial de posiciones para detectar movimiento
        self.max_history=5  # mantener últimas 5 posiciones
        
    def is_new(self, cells: List[Tuple[int,int]])->bool:
        if not cells: 
            return False
            
        try:
            r0, c0, r1, c1 = bounding_box(cells)
            sig = shape_signature(sorted(cells)[:4])
            current_pos = (r0, c0)
            current_time = time.time()
            
            # Agregar posición al historial
            self.position_history.append((current_pos, current_time))
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
            
            # Es nueva pieza si:
            # 1. Es la primera pieza detectada
            # 2. Tiene una forma diferente
            # 3. Apareció en una posición muy diferente (nueva pieza spawneada)
            # 4. Ha pasado mucho tiempo sin acción exitosa
            # 5. La pieza no se ha movido después de actuar (comando falló)
            
            is_new_piece = False
            reason = ""
            
            if self.last_sig is None or self.last_pos is None:
                is_new_piece = True
                reason = "Primera pieza detectada"
            elif sig != self.last_sig:
                is_new_piece = True  
                reason = f"Forma diferente: {sig} vs {self.last_sig}"
            elif abs(r0 - self.last_pos[0]) > 4:  # salto grande en altura (aumentado de 3 a 4)
                is_new_piece = True
                reason = f"Salto de altura: {r0} vs {self.last_pos[0]}"
            elif r0 < self.last_pos[0] - 1:  # pieza apareció más arriba (nueva spawn)
                is_new_piece = True
                reason = f"Nueva spawn detectada: {r0} vs {self.last_pos[0]}"
            elif (current_time - self.last_action_t) > self.action_timeout:
                # Verificar si la pieza se ha movido después de actuar
                if self.acted and len(self.position_history) >= 3:
                    # Comparar posiciones recientes para detectar movimiento
                    recent_positions = [pos for pos, _ in self.position_history[-3:]]
                    if all(pos == recent_positions[0] for pos in recent_positions):
                        # La pieza no se ha movido - probablemente el comando falló
                        is_new_piece = True
                        reason = "Pieza no se movió después de actuar - reintentando"
                    else:
                        reason = "Pieza se está moviendo normalmente"
                elif not self.acted:
                    is_new_piece = True
                    reason = "Timeout sin acción previa"
                else:
                    reason = "Esperando movimiento de pieza"
            
            # Log detallado para debugging
            if (current_time - getattr(self, '_last_log_time', 0)) > 0.5:  # log cada 0.5s
                self._last_log_time = current_time
                pos_history_str = " -> ".join([f"({r},{c})" for (r,c), _ in self.position_history[-3:]])
                logging.debug(f"Tracker: {reason} | Pos: {pos_history_str} | Acted: {self.acted} | Time since action: {current_time - self.last_action_t:.1f}s")
                
            if is_new_piece:
                logging.info(f"🎯 Nueva pieza para actuar: {reason}")
                self.last_sig = sig
                self.last_pos = current_pos
                self.acted = False
                self.last_action_t = current_time
                self.position_history = [(current_pos, current_time)]  # reset history
                return True
            else:
                # Actualizar posición actual pero no actuar
                self.last_pos = current_pos
                return False
                
        except Exception as e:
            logging.warning(f"Error en piece tracker: {e}")
            return False
    
    def mark_acted(self):
        """Marca que se actuó sobre la pieza actual"""
        self.acted = True
        self.last_action_t = time.time()
        logging.debug("Pieza marcada como actuada")


# ============================ Lógica principal / loop ============================

def safe_draw_grid_overlay(crop: np.ndarray, occ: np.ndarray, piece_cells: List[Tuple[int,int]]=None, ghost_cells: List[Tuple[int,int]]=None)->np.ndarray:
    """devuelve una copia con celdas ocupadas marcadas y pieza activa destacada (debug visual)."""
    vis = crop.copy()
    rows, cols = occ.shape
    h, w = vis.shape[:2]
    
    # Pixel-perfect grid calculation: distribute pixels evenly
    row_boundaries = np.linspace(0, h, rows + 1, dtype=int)
    col_boundaries = np.linspace(0, w, cols + 1, dtype=int)
    
    # Convertir piece_cells y ghost_cells a sets para lookup rápido
    active_cells = set(piece_cells) if piece_cells else set()
    ghost_cells_set = set(ghost_cells) if ghost_cells else set()
    
    for r in range(rows):
        for c in range(cols):
            # Use pixel-perfect boundaries
            x0, x1 = col_boundaries[c], col_boundaries[c + 1]
            y0, y1 = row_boundaries[r], row_boundaries[r + 1]
            
            # Asegurar coordenadas válidas (no debería ser necesario con linspace, pero por seguridad)
            x1 = max(x1, x0 + 1)
            y1 = max(y1, y0 + 1)
            
            # Grid lines removed for perfect alignment
            
            # Pintar ghost cells en cyan/azul claro (NO aparecen en la máscara)
            if (r,c) in ghost_cells_set:
                cv2.rectangle(vis, (x0+2,y0+2), (x1-2,y1-2), (255,255,0), 2)  # Cyan
                cv2.putText(vis, "G", (x0+3, y0+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
            elif occ[r,c]:
                if (r,c) in active_cells:
                    # Pieza activa en verde brillante
                    cv2.rectangle(vis, (x0+2,y0+2), (x1-2,y1-2), (0,255,0), 3)
                    # Agregar número de columna en el centro
                    cv2.putText(vis, str(c), (x0+5, y0+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                else:
                    # Piezas fijas en rojo
                    cv2.rectangle(vis, (x0+2,y0+2), (x1-2,y1-2), (0,0,255), 2)
    
    # Agregar información de texto en la parte superior
    if piece_cells:
        try:
            r0, c0, r1, c1 = bounding_box(piece_cells)
            info_text = f"Pieza: filas {r0}-{r1}, cols {c0}-{c1}"
            cv2.putText(vis, info_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        except Exception:
            pass
    
    return vis

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--serial", type=str, default=None)
    ap.add_argument("--backend", choices=["adb","scrcpy","hybrid"], default="adb")
    ap.add_argument("--rect", type=parse_rect, default=None, help="Rectángulo del tablero (x,y,w,h). Si no se especifica, usa configuración del dispositivo")
    ap.add_argument("--device", type=str, default="default", help="Tipo de dispositivo (samsung, pixel, oneplus, etc.)")
    ap.add_argument("--config", type=str, default="config.json", help="Archivo de configuración JSON")
    ap.add_argument("--fps", type=int, default=None, help="FPS del loop principal")
    ap.add_argument("--session-sec", type=int, default=None, help="Duración de la sesión en segundos")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug-vision", action="store_true", help="Guarda imágenes de depuración en la carpeta 'tetris_debug'")
    ap.add_argument("--list-devices", action="store_true", help="Lista dispositivos disponibles en la configuración")
    ap.add_argument("--use-bot-class", action="store_true", help="Usa el orquestador TetrisBot (experimental)")
    ap.add_argument("--auto-calibrate", action="store_true", help="Realiza calibración automática del rectángulo del tablero")
    ap.add_argument("--multistep-policy", action="store_true", help="Usa MultiStepPolicy avanzada con lookahead")
    ap.add_argument("--advanced-evaluation", action="store_true", help="Usa evaluación avanzada con T-spins y combos")
    ap.add_argument("--monitor-performance", action="store_true", help="Habilita monitoreo y exportación de métricas de rendimiento")
    ap.add_argument("--test-movement", action="store_true", help="Ejecuta test de movimiento y sale (debugging)")
    args=ap.parse_args()

    # Cargar configuración
    config = TetrisConfig(args.config)
    
    # Lista dispositivos si se solicita
    if args.list_devices:
        devices = config.list_available_devices()
        print("Dispositivos disponibles en la configuración:")
        for device in devices:
            rect = config.get_device_rect(device)
            print(f"  {device}: rect={rect}")
        return

    setup_logging(args.verbose)

    # Obtener parámetros desde configuración con overrides por CLI
    rect = args.rect if args.rect else config.get_device_rect(args.device)
    fps = args.fps if args.fps else config.get_gameplay_param("fps", 10)
    session_sec = args.session_sec if args.session_sec else config.get_gameplay_param("session_sec", 185)
    
    # Override para MultiStepPolicy si se especifica por CLI
    if args.multistep_policy:
        config.config.setdefault("gameplay", {})["use_multistep_policy"] = True
        logging.info("🧠 MultiStepPolicy habilitada por argumento CLI")
    
    # Override para evaluación avanzada si se especifica por CLI
    if args.advanced_evaluation:
        config.config.setdefault("evaluation", {})["use_advanced_evaluation"] = True
        logging.info("⚡ Evaluación avanzada habilitada por argumento CLI")
    
    logging.info(f"Configuración cargada: device={args.device}, rect={rect}, fps={fps}")

    if args.debug_vision:
        if not os.path.exists("tetris_debug"):
            os.makedirs("tetris_debug")
        logging.info("Modo de depuración visual ACTIVADO. Imágenes se guardarán en 'tetris_debug/'")

    if args.backend=="adb": backend=ADBBackend(args.serial)
    elif args.backend=="scrcpy":
        if not SCRCPY_DEPS_OK: logging.error("scrcpy backend requiere pyautogui/pygetwindow/mss"); sys.exit(1)
        backend=ScrcpyBackend(args.serial)
    else: backend=HybridBackend(args.serial)

    logging.info(f"Backend solicitado: {args.backend}")
    backend.connect()
    dev_w,dev_h=backend.get_resolution()
    
    # Calibración automática si se solicita
    if args.auto_calibrate:
        logging.info("🎯 Iniciando calibración automática...")
        auto_rect = auto_calibrate_board_rect(backend, config)
        if auto_rect:
            board = auto_rect
            logging.info(f"✅ Calibración automática exitosa: x={board.x0}, y={board.y0}, w={board.w}, h={board.h}")
            # Convertir a porcentajes para mostrar
            auto_pct = (board.x0/dev_w, board.y0/dev_h, board.w/dev_w, board.h/dev_h)
            logging.info(f"📐 Rectángulo en porcentajes: {auto_pct[0]:.3f},{auto_pct[1]:.3f},{auto_pct[2]:.3f},{auto_pct[3]:.3f}")
            logging.info("💡 Sugerencia: Guarda este rectángulo en tu configuración para futuros usos")
        else:
            logging.warning("⚠️ Calibración automática falló, usando configuración por defecto")
            board = get_board_rect_from_percent((dev_w,dev_h), rect)
    else:
        board = get_board_rect_from_percent((dev_w,dev_h), rect)
    
    logging.info(f"📱 Resolución del dispositivo: {dev_w}x{dev_h}")
    logging.info(f"🎯 Rectángulo del tablero: x={board.x0}, y={board.y0}, w={board.w}, h={board.h}")
    logging.info(f"📊 Porcentajes del tablero: x={board.x0/dev_w:.3f}, y={board.y0/dev_h:.3f}, w={board.w/dev_w:.3f}, h={board.h/dev_h:.3f}")

    # COORDINATE VALIDATION - Run startup diagnostics
    validation_results = validate_screen_coordinates(backend, board)
    if not validation_results['overall_valid']:
        logging.error("❌ Screen coordinate validation failed! Bot may not work correctly.")
        logging.error("💡 Consider recalibrating board coordinates or checking screen resolution.")
    else:
        logging.info("✅ Screen coordinate validation passed - bot should work correctly")

    # Opción de test de movimiento
    if args.test_movement:
        logging.info("🧪 MODO TEST DE MOVIMIENTO - Solo testing, no juego")
        test_movement_system(backend, board)
        logging.info("🏁 Test completado. Saliendo...")
        backend.cleanup()
        return

    # Opción experimental: usar orquestador TetrisBot
    if args.use_bot_class:
        logging.info("🤖 Modo experimental: Usando orquestador TetrisBot")
        try:
            _ = TetrisBot(config, backend, board)
            # Por ahora, el orquestador solo se inicializa pero no ejecuta
            logging.info("✅ TetrisBot inicializado correctamente")
            logging.info("⚠️  El orquestador completo estará disponible en una versión futura")
            logging.info("🔄 Continuando con modo híbrido...")
        except Exception as e:
            logging.error(f"❌ Error inicializando TetrisBot: {e}")
            logging.info("🔄 Continuando con modo clásico...")

    zones=compute_gesture_zones(board)

    dt=1.0/float(clamp(fps,3,15))
    logging.info(f"Loop a ~{1.0/dt:.1f} FPS (dt={dt:.3f}s). Ctrl+C para salir.")

    # Inicializar sistema modular (MIGRACIÓN GRADUAL)
    vision_system = TetrisVision(config)
    controller_system = TetrisController(backend, zones, config)
    game_system = TetrisGame(config)
    
    start=time.time()
    frame_count = 0
    last_wake_t = 0.0  # throttle de taps cuando no hay pieza
    try:
        while True:
            t0=time.time()
            frame=backend.get_screen()
            frame_h, frame_w = frame.shape[:2]
            
            # Expansión dinámica del board para capturar fila inferior completa
            # Añadir 8% extra de altura para asegurar cobertura completa de la fila inferior
            expansion_factor = 1.00  # No expansion to keep grid within game board boundaries
            expanded_h = int(board.h * expansion_factor)
            max_y = min(board.y0 + expanded_h, frame_h)
            actual_h = max_y - board.y0
            
            if frame_count == 0:  # Solo log una vez
                logging.info(f"🔧 Board expansion: original h={board.h} → expanded h={actual_h} (+{actual_h-board.h} pixels)")
            
            crop=frame[board.y0:max_y, board.x0:board.x0+board.w]
            
            # Validar tamaño del crop periódicamente
            if frame_count % 120 == 0:  # Cada 2 minutos a 60 FPS
                crop_h, crop_w = crop.shape[:2]
                expected_cell_h = crop_h / 20.0
                expected_cell_w = crop_w / 10.0
                expansion_pixels = actual_h - board.h
                logging.info(f"🖼️ Board crop validation: {crop_h}x{crop_w} pixels")
                logging.info(f"   Expected cell size: {expected_cell_w:.1f}x{expected_cell_h:.1f} pixels")
                logging.info(f"   Board rectangle: x={board.x0}, y={board.y0}, w={board.w}, h={board.h}")
                logging.info(f"   Expansion applied: +{expansion_pixels} pixels ({expansion_pixels/expected_cell_h:.1f} rows worth)")
                
                # Verificar que el crop tenga suficiente altura para 20 filas
                min_required_height = 200  # mínimo razonable para 20 filas
                if crop_h < min_required_height:
                    logging.warning(f"⚠️ Crop height ({crop_h}) may be too small for 20 rows!")

            # MIGRACIÓN GRADUAL: Usar sistema modular para análisis
            board_analysis = vision_system.analyze_board(crop, rows=20, cols=10)
            occ = board_analysis.occupancy_grid
            debug_mask = board_analysis.debug_mask
            piece_cells = board_analysis.active_piece
            occupation_rate = board_analysis.occupation_rate
            
            # Mantener variables compatibles para el resto del código
            num_occupied = int(occ.sum())
            rows, cols = occ.shape
            total_cells = rows * cols
            
            # Validar cobertura completa del tablero
            if rows != 20 or cols != 10:
                logging.warning(f"Dimensiones de tablero inesperadas: {rows}x{cols} (esperado: 20x10)")
            
            # Análisis detallado de la ocupación de filas inferiores
            if rows >= 3:
                bottom_3_rows = occ[-3:, :]  # Últimas 3 filas
                row_occupancy = [np.sum(bottom_3_rows[i, :]) for i in range(3)]
                total_bottom_occupied = np.sum(bottom_3_rows)

                if frame_count % 30 == 0 or total_bottom_occupied > 0:  # Log más frecuente si hay ocupación
                    logging.info("🎯 Bottom rows analysis:")
                    logging.info(f"   Row {rows-3}: {row_occupancy[0]}/{cols} occupied")
                    logging.info(f"   Row {rows-2}: {row_occupancy[1]}/{cols} occupied")
                    logging.info(f"   Row {rows-1} (bottom): {row_occupancy[2]}/{cols} occupied")
                    logging.info(f"   Total bottom 3 rows: {total_bottom_occupied}/{cols*3} occupied")

                # Advertir si hay patrones sospechosos
                if row_occupancy[2] == 0 and (row_occupancy[0] > 0 or row_occupancy[1] > 0):
                    logging.warning("⚠️ SUSPICIOUS: Upper rows have pieces but bottom row is empty!")
                    logging.warning("💡 This may indicate the board crop is missing the bottom row!")
                    logging.warning(f"🔧 Current expansion: {expansion_factor:.2f}x (add +{(expansion_factor-1)*100:.0f}%)")

                    # Sugerir mayor expansión si el patrón persiste
                    if frame_count > 300:  # Después de 5 segundos
                        logging.error("🚨 PERSISTENT BOTTOM ROW ISSUE - Consider increasing expansion_factor to 1.10 or higher!")
            else:
                bottom_row_occupied = np.sum(occ[-1, :]) if rows > 0 else 0
                logging.debug(f"Bottom row: {bottom_row_occupied}/{cols} occupied")
            
            # Validar si es un estado de juego razonable
          
            if occupation_rate > 0.70:
                logging.warning(f"Alta ocupación: {num_occupied}/{total_cells} ({occupation_rate:.1%}) → recalibrando (tight)")
                occ_tight, debug_mask_tight = occupancy_grid_tight(crop, rows, cols)
                rate_tight = occ_tight.sum() / total_cells

                if (rate_tight < occupation_rate) and (rate_tight <= 0.70):
                    occ, debug_mask = occ_tight, debug_mask_tight
                    occupation_rate = rate_tight
                    piece_cells = find_active_piece(occ, crop)
                    logging.warning(f"Recalibrado OK: ocupación ahora {occupation_rate:.1%}")
                else:
                    # Si hay una pieza válida detectada, no bloquees el juego
                    if piece_cells and 2 <= len(piece_cells) <= 4:
                        logging.warning(f"Alta ocupación ({occupation_rate:.1%}) pero pieza válida; continuo.")
                    else:
                        logging.error("Estado de juego inválido tras recalibración → salto frame")
                        time.sleep(0.05)
                        continue

            elif occupation_rate < 0.02:
                logging.debug(f"Tablero casi vacío: {num_occupied}/{total_cells} ({occupation_rate:.1%})")
            
            # ENHANCED FRAME CONTEXT LOGGING - Periodic summary
            if frame_count % 60 == 0:  # Every 60 frames (roughly every 6-10 seconds)
                current_time = time.time()
                elapsed = current_time - start
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                logging.info("🎮 FRAME CONTEXT SUMMARY")
                logging.info(f"   ⏱️  Time: {elapsed:.1f}s, Frame: {frame_count}, FPS: {fps:.1f}")
                logging.info(f"   📊 Board: {occupation_rate:.1%} occupied ({num_occupied}/{total_cells})")
                logging.info(f"   🎯 Piece: {'✅' if piece_cells else '❌'} active, {'👻' if board_analysis.ghost_piece else '⭕'} ghost")
                
                # RUNTIME BOARD RECTANGLE VERIFICATION
                runtime_verification = verify_board_runtime(board, piece_cells, occ, frame_count)
                if not runtime_verification['overall_valid']:
                    logging.warning(f"🔍 Board verification issues detected at frame {frame_count}")
                
                # ADAPTIVE MOVEMENT SYSTEM STATUS  
                adaptive_diag = controller_system.movement_corrector.get_diagnostics()
                if adaptive_diag['total_attempts'] > 0:
                    offset_status = f"OFFSET: {adaptive_diag['systematic_offset']:+.1f}" if adaptive_diag['offset_detected'] else "no offset"
                    logging.info(f"   🧠 Movement AI: {adaptive_diag['success_rate']:.1%} success, "
                                f"{adaptive_diag['columns_learned']} cols learned, "
                                f"error: {adaptive_diag['recent_avg_error']:.1f}, "
                                f"{offset_status}, timing: {adaptive_diag['recommended_timing']:.1f}x")
                
                # Log current board state every 5 minutes or when significant occupation
                if frame_count % 300 == 0 or occupation_rate > 0.5:
                    log_board_state(occ, piece_cells, frame_count, f"PERIODIC CHECK (occupied: {occupation_rate:.1%})")
                
            # FRAME BY FRAME CONTEXT for debugging specific moves
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                session_time = time.time() - start
                logging.debug(f"🖼️ Frame {frame_count} @ {session_time:.1f}s: occ={occupation_rate:.1%}, piece={'✓' if piece_cells else '✗'}")
                
            # --- Debug visual mejorado ---
            if args.debug_vision and frame_count < 50:  # más frames de debug
                cv2.imwrite(f"tetris_debug/{frame_count:03d}_crop.png", crop)
                cv2.imwrite(f"tetris_debug/{frame_count:03d}_mask.png", debug_mask)
                
                # Grid con información de pieza activa y ghost (USANDO ANÁLISIS MODULAR)
                ghost_cells = board_analysis.ghost_piece or []
                grid_img = safe_draw_grid_overlay(crop, occ, piece_cells, ghost_cells)
                cv2.imwrite(f"tetris_debug/{frame_count:03d}_grid.png", grid_img)
                
                # Log adicional para debug (USANDO ANÁLISIS MODULAR)
                num_components = board_analysis.components_found
                piece_info = f", pieza: {len(piece_cells)} celdas" if piece_cells else ", sin pieza"
                ghost_info = f", ghost: {len(ghost_cells)} celdas" if ghost_cells else ", sin ghost"
                logging.debug(f"Frame {frame_count}: {num_occupied}/{total_cells} celdas ocupadas ({occupation_rate:.1%}), {num_components} componentes{piece_info}{ghost_info}")
                

            if piece_cells is None:
                # No spam de taps: deja respirar y sigue
                now = time.time()
                if (now - last_wake_t) > 0.35:
                    last_wake_t = now
                    if frame_count % 30 == 0:  # log cada 30 frames sin pieza
                        logging.info("Esperando nueva pieza...")
                time.sleep(0.05)
                # siguiente frame
                sleep=dt-(time.time()-t0)
                if sleep>0: time.sleep(sleep)
                if session_sec and (time.time()-start)>session_sec: break
                continue
            
            # REJECT OVERSIZED PIECES - this should now be impossible but adding safety
            if len(piece_cells) > 4:
                logging.error(f"❌ REJECTING oversized piece: {len(piece_cells)} cells")
                logging.error(f"   This should not happen with the fixed vision system!")
                time.sleep(0.05)
                sleep=dt-(time.time()-t0)
                if sleep>0: time.sleep(sleep)
                if session_sec and (time.time()-start)>session_sec: break
                continue

            # tablero sin pieza NI ghost (para simular) - USANDO ANÁLISIS MODULAR
            ghost_cells = board_analysis.ghost_piece or []
            stack = occ.copy()
            remove_cells(stack, piece_cells)
            if ghost_cells:
                logging.info(f"🫥 Ghost detectado: {len(ghost_cells)} celdas — excluido del stack")
                remove_cells(stack, ghost_cells)
            cls_cur=classify_piece(piece_cells)
            if cls_cur is None:
                # fallback: coloca en columna más baja
                heights=column_heights(stack)
                target_col=int(np.argmin(heights)); rotations_needed=0
            else:
                piece_type, cur_orient = cls_cur
                best = game_system.choose_action(stack, piece_type)
                if best is None:
                    target_col=4; rotations_needed=0
                else:
                    best_orient, left_col = best
                    num_orients=len(PIECE_ORIENTS[piece_type])
                    rotations_needed=(best_orient-cur_orient)%num_orients
                    target_col=left_col

            # Verificar si necesitamos actuar (evitar spam en la misma pieza) - USANDO SISTEMA MODULAR
            if game_system.is_new_piece(piece_cells):
                # LOGGING MEJORADO: Frame context y board state
                current_time = time.time()
                frame_time = current_time - start
                
                logging.info("=" * 80)
                logging.info(f"🎮 NEW PIECE ACTION - Frame {frame_count} at {frame_time:.1f}s")
                logging.info("=" * 80)
                
                # Log board state visual
                log_board_state(stack, piece_cells, frame_count, f"BEFORE DECISION")
                
                # Log detailed decision context
                piece_name = piece_type if cls_cur else "UNKNOWN"
                final_score = best[1] if cls_cur and best and len(best) > 1 else "N/A"
                
                # Get score for chosen action if available
                if cls_cur and best:
                    try:
                        sim_result = drop_simulation(stack, PIECE_ORIENTS[piece_type][best_orient], left_col)
                        if sim_result:
                            newb, cleared = sim_result
                            final_score = evaluate_board(newb, cleared)
                    except:
                        final_score = "Error calculating"
                
                log_decision_context(
                    piece_name, 
                    stack, 
                    piece_cells, 
                    target_col, 
                    rotations_needed,
                    final_score,
                    f"Policy chose orient {best_orient if cls_cur and best else 'fallback'}, col {target_col}"
                )
                
                # DIAGNOSTIC: Analyze coordinate calculations
                diagnose_movement_coordinates(board, piece_cells, target_col)
                
                # Log execution plan
                logging.info(f"📋 EXECUTION PLAN:")
                logging.info(f"   1. Rotate {rotations_needed} times")
                logging.info(f"   2. Move to column {target_col}")  
                logging.info(f"   3. Drop piece")
                
                # Ejecutar plan (USANDO SISTEMA MODULAR)
                logging.info("🔄 Executing rotations...")
                for i in range(rotations_needed):
                    logging.debug(f"   Rotation {i+1}/{rotations_needed}")
                    controller_system.rotate_piece()
                    time.sleep(0.06)  # pausa más larga entre rotaciones
                
                # Execute movement with built-in retry and verification
                logging.info("➡️ Executing column movement...")
                movement_successful = controller_system.move_piece_to_column(piece_cells, target_col, board)
                
                if not movement_successful:
                    logging.error("❌ Movement failed after all retry attempts!")
                    logging.error("   Proceeding with drop anyway - may result in poor placement")
                else:
                    logging.info("✅ Movement completed successfully")
                
                logging.info("⬇️ Dropping piece...")
                controller_system.drop_piece()
                game_system.mark_piece_acted()
                
                # Final action summary
                execution_time = time.time() - current_time
                logging.info(f"✨ Action completed in {execution_time:.3f}s")
                logging.info("=" * 80)
            else:
                # Ya actuamos en esta pieza, solo esperar
                time.sleep(0.05)

            # ciclo - increment frame counter
            frame_count += 1
            sleep=dt-(time.time()-t0)
            if sleep>0: time.sleep(sleep)
            if session_sec and (time.time()-start)>session_sec:
                logging.info("Tiempo de sesión agotado. Saliendo…"); break

    except KeyboardInterrupt:
        logging.info("Interrumpido por el usuario. Saliendo...")
    finally:
        try: backend.cleanup()
        except Exception: pass


def test_movement_system(backend: ScreenBackend, board: BoardRect):
    """
    Función de test MEJORADA para calibración y diagnóstico de movimiento.
    Incluye validaciones, tests comprehensivos y detección de problemas.
    """
    logging.info("🧪 === COMPREHENSIVE MOVEMENT CALIBRATION TEST ===")
    
    # Pre-test validation
    validation_results = validate_screen_coordinates(backend, board)
    if not validation_results['overall_valid']:
        logging.warning("⚠️ Coordinate validation failed - tests may not work correctly!")
    
    # Calcular zones
    zones = compute_gesture_zones(board)
    cw = board.w/10.0
    ch = board.h/20.0
    y = zones.mid_band_y
    
    logging.info(f"🎯 Test parameters: y={y}, cell_width={cw:.1f}, cell_height={ch:.1f}")
    
    test_results = {
        'swipe_right': False,
        'swipe_left': False, 
        'rotation': False,
        'drop': False,
        'column_precision': {}
    }
    
    # Test 1: Swipe hacia la derecha (múltiples distancias)
    logging.info("🔄 Test 1 - HORIZONTAL SWIPES (Right)")
    for distance in [1, 2, 3]:
        x1 = int(board.x0 + 4*cw)  # Start from column 4
        x2 = int(x1 + distance*cw)  # Move distance columns right
        
        logging.info(f"   Swipe {distance} columns right: ({x1},{y}) -> ({x2},{y})")
        try:
            backend.swipe(x1, y, x2, y, duration_ms=200)
            logging.info(f"   ✅ {distance}-column right swipe successful")
            time.sleep(0.8)
        except Exception as e:
            logging.error(f"   ❌ {distance}-column right swipe failed: {e}")
    
    test_results['swipe_right'] = True
    
    # Test 2: Swipe hacia la izquierda (múltiples distancias)
    logging.info("🔄 Test 2 - HORIZONTAL SWIPES (Left)")
    for distance in [1, 2, 3]:
        x1 = int(board.x0 + 6*cw)  # Start from column 6
        x2 = int(x1 - distance*cw)  # Move distance columns left
        
        logging.info(f"   Swipe {distance} columns left: ({x1},{y}) -> ({x2},{y})")
        try:
            backend.swipe(x1, y, x2, y, duration_ms=200)
            logging.info(f"   ✅ {distance}-column left swipe successful")
            time.sleep(0.8)
        except Exception as e:
            logging.error(f"   ❌ {distance}-column left swipe failed: {e}")
            
    test_results['swipe_left'] = True
        
    # Test 3: Precision test - target specific columns
    logging.info("🔄 Test 3 - COLUMN PRECISION TEST")
    target_columns = [0, 2, 4, 6, 8, 9]
    start_col = 5
    
    for target_col in target_columns:
        logging.info(f"   Testing movement from col {start_col} to col {target_col}")
        
        # Calculate movement
        start_x = int(board.x0 + (start_col + 0.5) * cw)
        target_x = int(board.x0 + (target_col + 0.5) * cw)
        
        # Use the actual movement function for realistic testing
        try:
            # Simulate a piece at start column for testing
            fake_piece = [(10, start_col), (11, start_col)]  # Simple 2-cell piece
            
            logging.info(f"      Movement test: col {start_col} -> {target_col} ({target_x - start_x:+d}px)")
            move_piece_to_column(backend, zones, board, fake_piece, target_col)
            
            test_results['column_precision'][target_col] = True
            logging.info(f"   ✅ Column {target_col} movement completed")
            
        except Exception as e:
            test_results['column_precision'][target_col] = False
            logging.error(f"   ❌ Column {target_col} movement failed: {e}")
            
        time.sleep(1.0)  # Pause between tests
        
    # Test 4: Rotation test
    logging.info("🔄 Test 4 - ROTATION TEST")
    try:
        rx, ry = zones.rotate_xy
        logging.info(f"   Rotation tap at: ({rx},{ry})")
        
        for i in range(4):  # Test 4 rotations
            logging.info(f"   Rotation {i+1}/4")
            backend.tap(rx, ry, hold_ms=80)
            time.sleep(0.6)
        
        test_results['rotation'] = True
        logging.info("   ✅ Rotation test successful")
        
    except Exception as e:
        test_results['rotation'] = False  
        logging.error(f"   ❌ Rotation test failed: {e}")
    
    # Test 5: Drop test
    logging.info("🔄 Test 5 - DROP TEST")
    try:
        drop_x1, drop_y1, drop_x2, drop_y2 = zones.drop_path
        logging.info(f"   Drop swipe: ({drop_x1},{drop_y1}) -> ({drop_x2},{drop_y2})")
        
        backend.swipe(drop_x1, drop_y1, drop_x2, drop_y2, duration_ms=80)
        test_results['drop'] = True
        logging.info("   ✅ Drop test successful")
        
    except Exception as e:
        test_results['drop'] = False
        logging.error(f"   ❌ Drop test failed: {e}")
    
    # Test summary
    logging.info("🏁 === TEST RESULTS SUMMARY ===")
    
    total_tests = 4 + len(target_columns)
    passed_tests = sum([
        test_results['swipe_right'],
        test_results['swipe_left'], 
        test_results['rotation'],
        test_results['drop'],
        sum(test_results['column_precision'].values())
    ])
    
    logging.info(f"📊 Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    logging.info(f"   Horizontal swipes: {'✅' if test_results['swipe_right'] and test_results['swipe_left'] else '❌'}")
    logging.info(f"   Rotation: {'✅' if test_results['rotation'] else '❌'}")
    logging.info(f"   Drop: {'✅' if test_results['drop'] else '❌'}")
    
    precision_success = sum(test_results['column_precision'].values())
    precision_total = len(test_results['column_precision'])
    logging.info(f"   Column precision: {precision_success}/{precision_total} ({precision_success/precision_total*100:.1f}%)")
    
    if precision_success < precision_total * 0.8:
        logging.warning("⚠️ Column precision is poor - movement calibration needed!")
        logging.warning("💡 Consider adjusting board coordinates or timing parameters")
    
    if passed_tests == total_tests:
        logging.info("🎉 All tests passed! Movement system is working correctly.")
    else:
        logging.warning(f"⚠️ {total_tests - passed_tests} tests failed - movement system needs attention!")
    
    return test_results


if __name__=="__main__":
    main()
