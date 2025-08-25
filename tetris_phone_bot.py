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
            "gameplay": {"fps": 10, "session_sec": 185, "piece_tracker_timeout": 1.5, "max_component_size": 8}
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
            history_size=config.config.get("vision", {}).get("temporal_filter_history", 5),
            confidence_threshold=config.config.get("vision", {}).get("temporal_filter_threshold", 0.6)
        )
    
    def analyze_board(self, crop: np.ndarray, rows=20, cols=10, use_temporal_filter=True) -> BoardAnalysis:
        """Análisis completo del tablero en un frame"""
        # Análisis básico
        occ, debug_mask = occupancy_grid(crop, rows, cols)

        # Detección de pieza activa con filtrado temporal
        raw_piece_cells = find_active_piece(occ, crop)

        if use_temporal_filter:
            # Añadir detección al filtro temporal
            self.temporal_filter.add_detection(raw_piece_cells)
            # Obtener pieza filtrada
            piece_cells = self.temporal_filter.get_filtered_piece()
        else:
            piece_cells = raw_piece_cells

        # Detección de ghost usando la pieza filtrada
        ghost_cells = detect_ghost_component(crop, occ, piece_cells) if piece_cells else []

        if ghost_cells:
            # Quitar ghost de la grilla de ocupación
            remove_cells(occ, ghost_cells)
            # Limpiar también la máscara de depuración para reflejar la corrección
            for r, c in ghost_cells:
                y0, y1, x0, x1 = _cell_rect(r, c, (rows, cols), crop.shape)
                debug_mask[y0:y1, x0:x1] = 0

        num_occupied = int(occ.sum())
        total_cells = rows * cols
        occupation_rate = num_occupied / total_cells

        components = list_components(occ, max_component_size=8)
        components_found = len(components)

        return BoardAnalysis(
            occupancy_grid=occ,
            debug_mask=debug_mask,
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
        """Calcula consenso entre múltiples detecciones válidas"""
        if not valid_detections:
            return None
            
        # Por simplicidad, usar la detección más reciente como base
        # En futuras mejoras se podría promediar posiciones
        latest_detection = valid_detections[-1]
        return latest_detection['cells']
    
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

class TetrisController:
    """Maneja todas las acciones de control del juego"""
    
    def __init__(self, backend: 'ScreenBackend', zones: 'GestureZones', config: TetrisConfig):
        self.backend = backend
        self.zones = zones
        self.config = config
    
    def rotate_piece(self):
        """Wrapper para rotate_action - migración gradual"""
        rotate_action(self.backend, self.zones)
    
    def move_piece_to_column(self, piece_cells: List[Tuple[int,int]], target_col: int, board: 'BoardRect'):
        """Wrapper para move_piece_to_column - migración gradual"""
        move_piece_to_column(self.backend, self.zones, board, piece_cells, target_col)
    
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
        self.fps = config.get_gameplay_param("fps", 10)
        self.session_sec = config.get_gameplay_param("session_sec", 185)
        self.dt = 1.0/float(clamp(self.fps, 3, 15))
        
        logging.info(f"TetrisBot inicializado - FPS: {self.fps}, Sesión: {self.session_sec}s")
    
    def run_game_loop(self, debug_vision=False, max_debug_frames=50):
        """Ejecuta el bucle principal del juego - MIGRACIÓN FUTURA"""
        # Por ahora, este método está vacío
        # La migración del bucle principal será en una fase posterior
        pass

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
    def swipe(self,x1:int,y1:int,x2:int,y2:int,duration_ms:int=130):
        d=max(110,int(duration_ms))
        logging.debug(f"ADB swipe: ({x1},{y1}) -> ({x2},{y2}) dur={d}ms")
        if self._shell_try([["input","swipe",str(x1),str(y1),str(x2),str(y2),str(d)]]):
            logging.debug("ADB swipe exitoso (input swipe)")
            return
        if self._shell_try([["cmd","input","swipe",str(x1),str(y1),str(x2),str(y2),str(d)]]):
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


def list_components(occ: np.ndarray, max_component_size: int = 8) -> List[List[Tuple[int,int]]]:
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
    SAT_DELTA_MIN = 8.0    # ghost debe ser menos saturado
    V_DELTA_MIN   = 5.0    # y un poco más brillante
    MAX_GHOST_SIZE = 6

    comps = list_components(occ, max_component_size=12)
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

    SAT_DELTA_MIN = 8.0
    V_DELTA_MIN   = 5.0

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

    comps = list_components(occ, max_component_size=8)
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

def evaluate_surface_flatness(heights: np.ndarray) -> float:
    """Bonus por mantener superficie relativamente plana"""
    if len(heights) < 2:
        return 0
    
    # Calcular varianza de alturas (menor = más plano)
    height_variance = np.var(heights)
    
    # Bonus inversamente proporcional a la varianza
    flatness_bonus = max(0, 1_000_000 - height_variance * 50_000)
    
    return flatness_bonus

def evaluate_board(board: np.ndarray, lines_cleared:int)->float:
    # Sprint 3 min: recompensa brutal a limpiar ahora mismo
    reward = [0, 1_000_000, 3_000_000, 8_000_000, 20_000_000][lines_cleared]
    h=column_heights(board)
    holes=count_holes(board); bump=bumpiness(h)
    agg_h=int(np.sum(h)); max_h=int(np.max(h))
    return reward - (holes*1500 + agg_h*6 + bump*40 + max_h*20)

class OneStepPolicy:
    def choose(self, board_stack: np.ndarray, piece: str)->Optional[Tuple[int,int]]:
        best=None; best_score=-1e18
        for oi,shape in enumerate(PIECE_ORIENTS[piece]):
            width=max(c for _,c in shape)+1
            for left in range(0, 10-width+1):
                sim=drop_simulation(board_stack, shape, left)
                if sim is None: continue
                newb, cleared = sim
                score=evaluate_board(newb, cleared)
                if score>best_score:
                    best_score=score; best=(oi,left)
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
        
        # Evaluación multi-step
        best_action = None
        best_score = -1e18
        
        # Evaluar todas las posibles acciones para la pieza actual
        for oi, shape in enumerate(PIECE_ORIENTS[piece]):
            width = max(c for _, c in shape) + 1
            for left in range(0, 10 - width + 1):
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
                
                if total_score > best_score:
                    best_score = total_score
                    best_action = (oi, left)
        
        return best_action
    
    def _choose_single_step(self, board_stack: np.ndarray, piece: str) -> Optional[Tuple[int, int]]:
        """Implementación de un solo paso (igual que OneStepPolicy)"""
        best = None
        best_score = -1e18
        
        for oi, shape in enumerate(PIECE_ORIENTS[piece]):
            width = max(c for _, c in shape) + 1
            for left in range(0, 10 - width + 1):
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
            for left in range(0, 10 - width + 1):
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

@dataclass
class GestureZones:
    rotate_xy: Tuple[int,int]
    mid_band_y: int
    drop_path: Tuple[int,int,int,int]  # x,y1->x,y2

def compute_gesture_zones(board: BoardRect)->GestureZones:
    xc = board.x0 + board.w//2
    # Rotación arriba para no interferir con movimiento
    rotate = (xc, int(board.y0 + 0.10*board.h))
    # Movimiento en banda baja (evita activar rotación)
    mid_band_y = int(board.y0 + 0.70*board.h)
    # Hard drop profundo
    drop = (xc, int(board.y0 + 0.50*board.h), xc, int(board.y0 + 0.95*board.h))
    return GestureZones(rotate_xy=rotate, mid_band_y=mid_band_y, drop_path=drop)

def column_center_x(board: BoardRect, col: float)->int:
    cw=board.w/10.0
    return int(board.x0+(col+0.5)*cw)

def rotate_action(backend: ScreenBackend, zones: GestureZones):
    x,y=zones.rotate_xy
    tap_x, tap_y = x+jitter(3), y+jitter(3)
    hold_time = random.randint(80,120)
    logging.info(f"🔄 Ejecutando rotación en ({tap_x},{tap_y}) por {hold_time}ms")
    # Tap más largo y con menos jitter para mayor confiabilidad
    backend.tap(tap_x, tap_y, hold_ms=hold_time)
    logging.debug("Rotación completada")

def move_piece_to_column(backend: ScreenBackend, zones: GestureZones, board: BoardRect,
                         piece_cells: List[Tuple[int,int]], target_col:int):
    """Mueve con swipes más grandes y cálculo de posición mejorado."""
    try:
        # Usar bounding box en lugar de promedio para mejor precisión
        r0, c0, r1, c1 = bounding_box(piece_cells)
        cur_col = c0  # usar columna izquierda como referencia
        logging.debug(f"Pieza actual en columnas {c0}-{c1}, moviendo hacia columna {target_col}")
    except Exception:
        cur_col = 5
        logging.warning("Error calculando posición actual, usando columna 5 por defecto")
    
    delta = int(target_col - cur_col)
    if delta == 0: 
        logging.debug("Ya en posición correcta, no se necesita movimiento")
        return
        
    steps = abs(delta)
    dir_sign = 1 if delta > 0 else -1
    y = zones.mid_band_y + jitter(4)
    cw = board.w/10.0
    dx = int(1.1*cw) * dir_sign  # aumentado de 0.7 a 1.1 celdas
    x = int(board.x0 + (cur_col + 0.5)*cw)
    
    logging.info(f"⬅️➡️ Moviendo {steps} pasos hacia {'derecha' if dir_sign > 0 else 'izquierda'} (delta={delta})")
    for i in range(steps):
        duration = random.randint(70,110)  # duración ligeramente mayor
        swipe_x1, swipe_x2 = int(x), int(x + dx)
        logging.debug(f"  Swipe {i+1}/{steps}: ({swipe_x1},{y}) -> ({swipe_x2},{y}) en {duration}ms")
        backend.swipe(swipe_x1, y, swipe_x2, y, duration_ms=duration)
        x += dx
        time.sleep(0.025)  # pausa ligeramente mayor
    logging.debug(f"Movimiento completado ({steps} swipes)")

def drop_action(backend: ScreenBackend, zones: GestureZones):
    x1,y1,x2,y2=zones.drop_path
    drop_x1, drop_y1 = x1+jitter(5), y1+jitter(5)
    drop_x2, drop_y2 = x2+jitter(5), y2+jitter(5)
    duration = random.randint(90,130)
    logging.info(f"📧 Ejecutando hard drop: ({drop_x1},{drop_y1}) -> ({drop_x2},{drop_y2}) en {duration}ms")
    backend.swipe(drop_x1, drop_y1, drop_x2, drop_y2, duration_ms=duration)
    logging.debug("Hard drop completado")


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
    
    logging.info(f"Rect del pozo: x={board.x0}, y={board.y0}, w={board.w}, h={board.h}")

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
                
                frame_count += 1

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
                logging.info(f"Nueva pieza detectada: {piece_type if cls_cur else 'desconocida'}, rotaciones={rotations_needed}, columna objetivo={target_col}")
                
                # Ejecutar plan (USANDO SISTEMA MODULAR)
                for i in range(rotations_needed):
                    controller_system.rotate_piece()
                    time.sleep(0.06)  # pausa más larga entre rotaciones
                
                controller_system.move_piece_to_column(piece_cells, target_col, board)
                time.sleep(0.04)
                
                # Verificar posición antes del drop
                time.sleep(0.08)
                verification_frame = backend.get_screen()
                # Aplicar la misma expansión al crop de verificación
                verification_frame_h = verification_frame.shape[0]
                verification_max_y = min(board.y0 + expanded_h, verification_frame_h)
                verification_crop = verification_frame[board.y0:verification_max_y, board.x0:board.x0+board.w]
                verification_occ, _ = occupancy_grid(verification_crop, rows=20, cols=10)
                verification_piece = find_active_piece(verification_occ, verification_crop)
                
                if verification_piece:
                    try:
                        _, v_c0, _, v_c1 = bounding_box(verification_piece)
                        actual_pos = v_c0
                        if abs(actual_pos - target_col) <= 1:  # tolerancia de 1 columna
                            logging.debug(f"Posición verificada correcta: {actual_pos} ≈ {target_col}")
                        else:
                            logging.warning(f"Posición incorrecta: objetivo={target_col}, actual={actual_pos}")
                    except Exception:
                        logging.warning("Error verificando posición")
                
                controller_system.drop_piece()
                game_system.mark_piece_acted()
            else:
                # Ya actuamos en esta pieza, solo esperar
                time.sleep(0.05)

            # ciclo
            sleep=dt-(time.time()-t0)
            if sleep>0: time.sleep(sleep)
            if session_sec and (time.time()-start)>session_sec:
                logging.info("Tiempo de sesión agotado. Saliendo…"); break

    except KeyboardInterrupt:
        logging.info("Interrumpido por el usuario. Saliendo...")
    finally:
        try: backend.cleanup()
        except Exception: pass


if __name__=="__main__":
    main()