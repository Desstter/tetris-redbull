import logging
from typing import List, Optional, Tuple

import numpy as np

from tetris_phone_bot import (
    BOARD_ROWS,
    BOARD_COLS,
    BoardAnalysis,
    TemporalFilter,
    TetrisConfig,
    detect_pieces_multilayer,
    occupancy_grid,
    occupancy_grid_multilayer,
    extract_components_by_type,
)


class TetrisVision:
    """Maneja todo el análisis visual del tablero de Tetris"""

    def __init__(self, config: TetrisConfig):
        self.config = config
        self.temporal_filter = TemporalFilter(
            history_size=config.config.get("vision", {}).get("temporal_filter_history", 7),
            confidence_threshold=config.config.get("vision", {}).get("temporal_filter_threshold", 0.7),
        )

    def _detect_components(self, crop: np.ndarray, rows: int, cols: int):
        """Detección primaria de piezas y grillas de ocupación"""
        raw_piece_cells, raw_ghost_cells, debug_info = detect_pieces_multilayer(crop, rows, cols)
        occupancy_grid(crop, rows, cols, mode="normal")
        _, active_grid, _, debug_mask = occupancy_grid_multilayer(crop, rows, cols)
        occ = active_grid.copy()
        debug_info["debug_mask"] = debug_mask
        return occ, raw_piece_cells, raw_ghost_cells, debug_info

    def _apply_temporal_filter(
        self,
        raw_piece_cells,
        raw_ghost_cells,
        crop: np.ndarray,
        rows: int,
        cols: int,
        use_temporal_filter: bool = True,
    ):
        """Aplica filtrado temporal para estabilizar la pieza detectada"""
        piece_cells = raw_piece_cells
        ghost_cells = raw_ghost_cells

        if use_temporal_filter and raw_piece_cells:
            self.temporal_filter.add_detection(raw_piece_cells)
            piece_cells = self.temporal_filter.get_filtered_piece()
            if piece_cells != raw_piece_cells and piece_cells:
                _, ghost_cells, _ = detect_pieces_multilayer(crop, rows, cols)

        return piece_cells, ghost_cells

    def _log_analysis(
        self,
        occ: np.ndarray,
        piece_cells,
        ghost_cells,
        rows: int,
        cols: int,
        use_temporal_filter: bool,
        raw_piece_cells,
    ):
        """Calcula estadísticas y registra mensajes de diagnóstico"""
        if piece_cells:
            for r, c in piece_cells:
                if 0 <= r < rows and 0 <= c < cols:
                    occ[r, c] = False
            for r, c in piece_cells:
                if 0 <= r < rows and 0 <= c < cols:
                    occ[r, c] = True

        num_occupied = int(occ.sum())
        total_cells = rows * cols
        occupation_rate = num_occupied / total_cells

        components = extract_components_by_type(occ, np.zeros_like(occ))[0]
        components_found = len(components)

        method_info = "multilayer + temporal" if use_temporal_filter else "multilayer"
        stability_score = self.temporal_filter.get_stability_score() if use_temporal_filter else 1.0

        logging.debug(f"👁️ VISION ANALYSIS ({method_info}):")
        logging.debug(
            f"   🔍 Grid analysis: {num_occupied}/{total_cells} occupied ({occupation_rate:.1%})"
        )
        logging.debug(f"   🧩 Components found: {components_found}")
        logging.debug(
            f"   🎯 Active piece: {len(piece_cells) if piece_cells else 0} cells detected"
        )
        logging.debug(
            f"   👻 Ghost piece: {len(ghost_cells) if ghost_cells else 0} cells detected"
        )
        if use_temporal_filter:
            logging.debug(f"   🎚️ Temporal stability: {stability_score:.2f}")
            if raw_piece_cells != piece_cells:
                logging.debug(
                    f"   🔄 Temporal filter active: raw {len(raw_piece_cells) if raw_piece_cells else 0} -> filtered {len(piece_cells) if piece_cells else 0}"
                )

        if piece_cells:
            try:
                r_min = min(r for r, _ in piece_cells)
                r_max = max(r for r, _ in piece_cells)
                c_min = min(c for _, c in piece_cells)
                c_max = max(c for _, c in piece_cells)
                piece_height = r_max - r_min + 1
                piece_width = c_max - c_min + 1
                logging.info(
                    f"🎯 Piece detected ({method_info}): {len(piece_cells)} cells at rows {r_min}-{r_max}, cols {c_min}-{c_max} ({piece_width}x{piece_height})"
                )
            except ValueError:
                logging.info(
                    f"🎯 Piece detected ({method_info}): {len(piece_cells)} cells (position unknown)"
                )

        if ghost_cells:
            try:
                r_min = min(r for r, _ in ghost_cells)
                r_max = max(r for r, _ in ghost_cells)
                c_min = min(c for _, c in ghost_cells)
                c_max = max(c for _, c in ghost_cells)
                logging.info(
                    f"👻 Ghost detected ({method_info}): {len(ghost_cells)} cells at rows {r_min}-{r_max}, cols {c_min}-{c_max}"
                )
            except ValueError:
                logging.info(
                    f"👻 Ghost detected ({method_info}): {len(ghost_cells)} cells"
                )

        if piece_cells and len(piece_cells) > 4:
            logging.error(
                f"❌ CRITICAL: Oversized piece detected: {len(piece_cells)} cells (max allowed: 4)"
            )
            logging.error(f"   Piece cells: {piece_cells}")
            logging.error(
                "   This indicates vision system malfunction - piece will be rejected"
            )
        elif piece_cells and len(piece_cells) == 1:
            logging.warning(
                "⚠️ Single cell piece detected - possibly noise or I-piece end"
            )
        elif piece_cells and len(piece_cells) in [2, 3, 4]:
            logging.debug(f"✅ Valid piece size: {len(piece_cells)} cells")

        if occupation_rate > 0.8:
            logging.warning(
                f"⚠️ Very high board occupation: {occupation_rate:.1%} - may indicate detection issues"
            )
        if components_found > 15:
            logging.warning(
                f"⚠️ High component count: {components_found} - possible noise in detection"
            )

        if piece_cells and ghost_cells:
            if len(set(piece_cells) & set(ghost_cells)) > 0:
                logging.warning(
                    "⚠️ Piece and ghost overlap detected - may indicate detection confusion"
                )

        return occupation_rate, components_found

    def analyze_board(
        self,
        crop: np.ndarray,
        rows: int = BOARD_ROWS,
        cols: int = BOARD_COLS,
        use_temporal_filter: bool = True,
    ) -> BoardAnalysis:
        """Orquesta el análisis de tablero usando funciones auxiliares"""
        occ, raw_piece_cells, raw_ghost_cells, debug_info = self._detect_components(
            crop, rows, cols
        )
        piece_cells, ghost_cells = self._apply_temporal_filter(
            raw_piece_cells, raw_ghost_cells, crop, rows, cols, use_temporal_filter
        )
        occupation_rate, components_found = self._log_analysis(
            occ, piece_cells, ghost_cells, rows, cols, use_temporal_filter, raw_piece_cells
        )
        return BoardAnalysis(
            occupancy_grid=occ,
            debug_mask=debug_info["debug_mask"],
            active_piece=piece_cells,
            ghost_piece=ghost_cells,
            occupation_rate=occupation_rate,
            components_found=components_found,
        )
