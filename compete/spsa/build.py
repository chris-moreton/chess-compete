"""
Engine building utilities for SPSA tuning.

Modifies engine_constants.rs with parameter values and compiles the engine.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

# Default path to rusty-rival source (relative to chess-compete root)
DEFAULT_RUSTY_RIVAL_PATH = "../rusty-rival"

# Mapping from params.toml names to engine_constants.rs patterns
# Format: param_name -> (regex_pattern, replacement_template)
# The template uses {value} as placeholder
PARAM_MAPPINGS = {
    # ==========================================================================
    # SEARCH PARAMETERS
    # ==========================================================================
    'beta_prune_margin_per_depth': (
        r'pub const BETA_PRUNE_MARGIN_PER_DEPTH: Score = \d+;',
        'pub const BETA_PRUNE_MARGIN_PER_DEPTH: Score = {value};'
    ),
    'beta_prune_max_depth': (
        r'pub const BETA_PRUNE_MAX_DEPTH: u8 = \d+;',
        'pub const BETA_PRUNE_MAX_DEPTH: u8 = {value};'
    ),
    'null_move_reduce_depth_base': (
        r'pub const NULL_MOVE_REDUCE_DEPTH_BASE: u8 = \d+;',
        'pub const NULL_MOVE_REDUCE_DEPTH_BASE: u8 = {value};'
    ),
    'null_move_min_depth': (
        r'pub const NULL_MOVE_MIN_DEPTH: u8 = \d+;',
        'pub const NULL_MOVE_MIN_DEPTH: u8 = {value};'
    ),
    'see_prune_margin': (
        r'pub const SEE_PRUNE_MARGIN: Score = \d+;',
        'pub const SEE_PRUNE_MARGIN: Score = {value};'
    ),
    'see_prune_max_depth': (
        r'pub const SEE_PRUNE_MAX_DEPTH: u8 = \d+;',
        'pub const SEE_PRUNE_MAX_DEPTH: u8 = {value};'
    ),
    'threat_extension_margin': (
        r'pub const THREAT_EXTENSION_MARGIN: Score = \d+;',
        'pub const THREAT_EXTENSION_MARGIN: Score = {value};'
    ),
    # Probcut parameters
    'probcut_min_depth': (
        r'pub const PROBCUT_MIN_DEPTH: u8 = \d+;',
        'pub const PROBCUT_MIN_DEPTH: u8 = {value};'
    ),
    'probcut_margin': (
        r'pub const PROBCUT_MARGIN: Score = \d+;',
        'pub const PROBCUT_MARGIN: Score = {value};'
    ),
    'probcut_depth_reduction': (
        r'pub const PROBCUT_DEPTH_REDUCTION: u8 = \d+;',
        'pub const PROBCUT_DEPTH_REDUCTION: u8 = {value};'
    ),
    # Multi-cut parameters
    'multicut_min_depth': (
        r'pub const MULTICUT_MIN_DEPTH: u8 = \d+;',
        'pub const MULTICUT_MIN_DEPTH: u8 = {value};'
    ),
    'multicut_depth_reduction': (
        r'pub const MULTICUT_DEPTH_REDUCTION: u8 = \d+;',
        'pub const MULTICUT_DEPTH_REDUCTION: u8 = {value};'
    ),
    'multicut_moves_to_try': (
        r'pub const MULTICUT_MOVES_TO_TRY: u8 = \d+;',
        'pub const MULTICUT_MOVES_TO_TRY: u8 = {value};'
    ),
    'multicut_required_cutoffs': (
        r'pub const MULTICUT_REQUIRED_CUTOFFS: u8 = \d+;',
        'pub const MULTICUT_REQUIRED_CUTOFFS: u8 = {value};'
    ),
    # ==========================================================================
    # SINGULAR EXTENSION PARAMETERS
    # ==========================================================================
    'singular_extension_min_depth': (
        r'pub const SINGULAR_EXTENSION_MIN_DEPTH: u8 = \d+;',
        'pub const SINGULAR_EXTENSION_MIN_DEPTH: u8 = {value};'
    ),
    'singular_extension_depth_margin': (
        r'pub const SINGULAR_EXTENSION_DEPTH_MARGIN: u8 = \d+;',
        'pub const SINGULAR_EXTENSION_DEPTH_MARGIN: u8 = {value};'
    ),
    'singular_extension_depth_reduction': (
        r'pub const SINGULAR_EXTENSION_DEPTH_REDUCTION: u8 = \d+;',
        'pub const SINGULAR_EXTENSION_DEPTH_REDUCTION: u8 = {value};'
    ),
    'singular_extension_margin_multiplier': (
        r'pub const SINGULAR_EXTENSION_MARGIN_MULTIPLIER: Score = \d+;',
        'pub const SINGULAR_EXTENSION_MARGIN_MULTIPLIER: Score = {value};'
    ),
    # ==========================================================================
    # MOVE ORDERING PARAMETERS
    # ==========================================================================
    'move_score_mate_killer': (
        r'pub const MOVE_SCORE_MATE_KILLER: Score = \d+;',
        'pub const MOVE_SCORE_MATE_KILLER: Score = {value};'
    ),
    'move_score_killer_1': (
        r'pub const MOVE_SCORE_KILLER_1: Score = \d+;',
        'pub const MOVE_SCORE_KILLER_1: Score = {value};'
    ),
    'move_score_killer_2': (
        r'pub const MOVE_SCORE_KILLER_2: Score = \d+;',
        'pub const MOVE_SCORE_KILLER_2: Score = {value};'
    ),
    'move_score_history_max': (
        r'pub const MOVE_SCORE_HISTORY_MAX: Score = \d+;',
        'pub const MOVE_SCORE_HISTORY_MAX: Score = {value};'
    ),
    'move_score_distant_killer_1': (
        r'pub const MOVE_SCORE_DISTANT_KILLER_1: Score = \d+;',
        'pub const MOVE_SCORE_DISTANT_KILLER_1: Score = {value};'
    ),
    'move_score_distant_killer_2': (
        r'pub const MOVE_SCORE_DISTANT_KILLER_2: Score = \d+;',
        'pub const MOVE_SCORE_DISTANT_KILLER_2: Score = {value};'
    ),
    'move_score_countermove': (
        r'pub const MOVE_SCORE_COUNTERMOVE: Score = \d+;',
        'pub const MOVE_SCORE_COUNTERMOVE: Score = {value};'
    ),
    'move_score_pawn_push_7th': (
        r'pub const MOVE_SCORE_PAWN_PUSH_7TH: Score = \d+;',
        'pub const MOVE_SCORE_PAWN_PUSH_7TH: Score = {value};'
    ),
    'move_score_pawn_push_6th': (
        r'pub const MOVE_SCORE_PAWN_PUSH_6TH: Score = \d+;',
        'pub const MOVE_SCORE_PAWN_PUSH_6TH: Score = {value};'
    ),
    'countermove_history_divisor': (
        r'pub const COUNTERMOVE_HISTORY_DIVISOR: i32 = \d+;',
        'pub const COUNTERMOVE_HISTORY_DIVISOR: i32 = {value};'
    ),
    'followup_history_divisor': (
        r'pub const FOLLOWUP_HISTORY_DIVISOR: i32 = \d+;',
        'pub const FOLLOWUP_HISTORY_DIVISOR: i32 = {value};'
    ),
    'capture_history_divisor': (
        r'pub const CAPTURE_HISTORY_DIVISOR: i32 = \d+;',
        'pub const CAPTURE_HISTORY_DIVISOR: i32 = {value};'
    ),
    # ==========================================================================
    # LMR HISTORY PARAMETERS
    # ==========================================================================
    'lmr_history_good_divisor': (
        r'pub const LMR_HISTORY_GOOD_DIVISOR: i32 = \d+;',
        'pub const LMR_HISTORY_GOOD_DIVISOR: i32 = {value};'
    ),
    'lmr_history_bad_divisor': (
        r'pub const LMR_HISTORY_BAD_DIVISOR: i32 = \d+;',
        'pub const LMR_HISTORY_BAD_DIVISOR: i32 = {value};'
    ),
    'lmr_continuation_good_threshold': (
        r'pub const LMR_CONTINUATION_GOOD_THRESHOLD: i32 = \d+;',
        'pub const LMR_CONTINUATION_GOOD_THRESHOLD: i32 = {value};'
    ),
    'lmr_continuation_bad_threshold': (
        r'pub const LMR_CONTINUATION_BAD_THRESHOLD: i32 = -\d+;',
        'pub const LMR_CONTINUATION_BAD_THRESHOLD: i32 = {value};'
    ),
    # ==========================================================================
    # SEARCH ADDITIONS
    # ==========================================================================
    'lmr_legal_moves_before_attempt': (
        r'pub const LMR_LEGAL_MOVES_BEFORE_ATTEMPT: u8 = \d+;',
        'pub const LMR_LEGAL_MOVES_BEFORE_ATTEMPT: u8 = {value};'
    ),
    'lmr_min_depth': (
        r'pub const LMR_MIN_DEPTH: u8 = \d+;',
        'pub const LMR_MIN_DEPTH: u8 = {value};'
    ),
    'iid_min_depth': (
        r'pub const IID_MIN_DEPTH: u8 = \d+;',
        'pub const IID_MIN_DEPTH: u8 = {value};'
    ),
    'iid_reduce_depth': (
        r'pub const IID_REDUCE_DEPTH: u8 = \d+;',
        'pub const IID_REDUCE_DEPTH: u8 = {value};'
    ),
    # ==========================================================================
    # EVALUATION PARAMETERS
    # ==========================================================================
    # Rook file bonuses
    'rook_open_file_bonus': (
        r'pub const ROOK_OPEN_FILE_BONUS: Score = \d+;',
        'pub const ROOK_OPEN_FILE_BONUS: Score = {value};'
    ),
    'rook_semi_open_file_bonus': (
        r'pub const ROOK_SEMI_OPEN_FILE_BONUS: Score = \d+;',
        'pub const ROOK_SEMI_OPEN_FILE_BONUS: Score = {value};'
    ),
    # Knight outposts
    'value_knight_outpost': (
        r'pub const VALUE_KNIGHT_OUTPOST: Score = \d+;',
        'pub const VALUE_KNIGHT_OUTPOST: Score = {value};'
    ),
    # Passed pawn bonuses
    'value_rook_behind_passed_pawn': (
        r'pub const VALUE_ROOK_BEHIND_PASSED_PAWN: Score = \d+;',
        'pub const VALUE_ROOK_BEHIND_PASSED_PAWN: Score = {value};'
    ),
    'value_guarded_passed_pawn': (
        r'pub const VALUE_GUARDED_PASSED_PAWN: Score = \d+;',
        'pub const VALUE_GUARDED_PASSED_PAWN: Score = {value};'
    ),
    # Pawn structure penalties
    'doubled_pawn_penalty': (
        r'pub const DOUBLED_PAWN_PENALTY: Score = \d+;',
        'pub const DOUBLED_PAWN_PENALTY: Score = {value};'
    ),
    'isolated_pawn_penalty': (
        r'pub const ISOLATED_PAWN_PENALTY: Score = \d+;',
        'pub const ISOLATED_PAWN_PENALTY: Score = {value};'
    ),
    'value_backward_pawn_penalty': (
        r'pub const VALUE_BACKWARD_PAWN_PENALTY: Score = \d+;',
        'pub const VALUE_BACKWARD_PAWN_PENALTY: Score = {value};'
    ),
    # Bishop pair bonus
    'value_bishop_pair': (
        r'pub const VALUE_BISHOP_PAIR: Score = \d+;',
        'pub const VALUE_BISHOP_PAIR: Score = {value};'
    ),
    # Space evaluation
    'space_bonus_per_square': (
        r'pub const SPACE_BONUS_PER_SQUARE: Score = \d+;',
        'pub const SPACE_BONUS_PER_SQUARE: Score = {value};'
    ),
    # Rooks on same file / seventh rank
    'value_rooks_on_same_file': (
        r'pub const VALUE_ROOKS_ON_SAME_FILE: Score = \d+;',
        'pub const VALUE_ROOKS_ON_SAME_FILE: Score = {value};'
    ),
    'rooks_on_seventh_rank_bonus': (
        r'pub const ROOKS_ON_SEVENTH_RANK_BONUS: Score = \d+;',
        'pub const ROOKS_ON_SEVENTH_RANK_BONUS: Score = {value};'
    ),
    # Trapped pieces
    'trapped_bishop_penalty': (
        r'pub const TRAPPED_BISHOP_PENALTY: Score = \d+;',
        'pub const TRAPPED_BISHOP_PENALTY: Score = {value};'
    ),
    'trapped_rook_penalty': (
        r'pub const TRAPPED_ROOK_PENALTY: Score = \d+;',
        'pub const TRAPPED_ROOK_PENALTY: Score = {value};'
    ),
    # ==========================================================================
    # MOBILITY CURVE PARAMETERS
    # ==========================================================================
    'bishop_mobility_base': (
        r'pub const BISHOP_MOBILITY_BASE: Score = -?\d+;',
        'pub const BISHOP_MOBILITY_BASE: Score = {value};'
    ),
    'bishop_mobility_scale_x100': (
        r'pub const BISHOP_MOBILITY_SCALE_X100: i32 = \d+;',
        'pub const BISHOP_MOBILITY_SCALE_X100: i32 = {value};'
    ),
    'queen_mobility_base': (
        r'pub const QUEEN_MOBILITY_BASE: Score = -?\d+;',
        'pub const QUEEN_MOBILITY_BASE: Score = {value};'
    ),
    'queen_mobility_scale_x100': (
        r'pub const QUEEN_MOBILITY_SCALE_X100: i32 = \d+;',
        'pub const QUEEN_MOBILITY_SCALE_X100: i32 = {value};'
    ),
    # ==========================================================================
    # PIECE ACTIVITY PARAMETERS
    # ==========================================================================
    'value_bishop_pair_fewer_pawns_bonus': (
        r'pub const VALUE_BISHOP_PAIR_FEWER_PAWNS_BONUS: Score = \d+;',
        'pub const VALUE_BISHOP_PAIR_FEWER_PAWNS_BONUS: Score = {value};'
    ),
    'bishop_knight_imbalance_bonus': (
        r'pub const BISHOP_KNIGHT_IMBALANCE_BONUS: Score = \d+;',
        'pub const BISHOP_KNIGHT_IMBALANCE_BONUS: Score = {value};'
    ),
    'knight_attacks_pawn_general_bonus': (
        r'pub const KNIGHT_ATTACKS_PAWN_GENERAL_BONUS: Score = \d+;',
        'pub const KNIGHT_ATTACKS_PAWN_GENERAL_BONUS: Score = {value};'
    ),
    'knight_fork_threat_score': (
        r'pub const KNIGHT_FORK_THREAT_SCORE: Score = \d+;',
        'pub const KNIGHT_FORK_THREAT_SCORE: Score = {value};'
    ),
    # ==========================================================================
    # KING PLAY PARAMETERS
    # ==========================================================================
    'king_threat_bonus_knight': (
        r'pub const KING_THREAT_BONUS_KNIGHT: Score = \d+;',
        'pub const KING_THREAT_BONUS_KNIGHT: Score = {value};'
    ),
    'king_threat_bonus_queen': (
        r'pub const KING_THREAT_BONUS_QUEEN: Score = \d+;',
        'pub const KING_THREAT_BONUS_QUEEN: Score = {value};'
    ),
    'king_threat_bonus_bishop': (
        r'pub const KING_THREAT_BONUS_BISHOP: Score = \d+;',
        'pub const KING_THREAT_BONUS_BISHOP: Score = {value};'
    ),
    'king_threat_bonus_rook': (
        r'pub const KING_THREAT_BONUS_ROOK: Score = \d+;',
        'pub const KING_THREAT_BONUS_ROOK: Score = {value};'
    ),
    'value_king_attacks_minor': (
        r'pub const VALUE_KING_ATTACKS_MINOR: Score = \d+;',
        'pub const VALUE_KING_ATTACKS_MINOR: Score = {value};'
    ),
    'value_king_attacks_rook': (
        r'pub const VALUE_KING_ATTACKS_ROOK: Score = \d+;',
        'pub const VALUE_KING_ATTACKS_ROOK: Score = {value};'
    ),
    'value_king_mobility': (
        r'pub const VALUE_KING_MOBILITY: Score = \d+;',
        'pub const VALUE_KING_MOBILITY: Score = {value};'
    ),
    'value_king_cannot_catch_pawn': (
        r'pub const VALUE_KING_CANNOT_CATCH_PAWN: Score = \d+;',
        'pub const VALUE_KING_CANNOT_CATCH_PAWN: Score = {value};'
    ),
    'value_king_cannot_catch_pawn_pieces_remain': (
        r'pub const VALUE_KING_CANNOT_CATCH_PAWN_PIECES_REMAIN: Score = \d+;',
        'pub const VALUE_KING_CANNOT_CATCH_PAWN_PIECES_REMAIN: Score = {value};'
    ),
    'value_king_distance_passed_pawn_multiplier': (
        r'pub const VALUE_KING_DISTANCE_PASSED_PAWN_MULTIPLIER: Score = \d+;',
        'pub const VALUE_KING_DISTANCE_PASSED_PAWN_MULTIPLIER: Score = {value};'
    ),
    'value_king_supports_passed_pawn': (
        r'pub const VALUE_KING_SUPPORTS_PASSED_PAWN: Score = \d+;',
        'pub const VALUE_KING_SUPPORTS_PASSED_PAWN: Score = {value};'
    ),
    # ==========================================================================
    # PASSED PAWN BLOCKER PARAMETERS
    # ==========================================================================
    'blocked_passed_pawn_penalty': (
        r'pub const BLOCKED_PASSED_PAWN_PENALTY: Score = \d+;',
        'pub const BLOCKED_PASSED_PAWN_PENALTY: Score = {value};'
    ),
    'knight_blockade_penalty': (
        r'pub const KNIGHT_BLOCKADE_PENALTY: Score = \d+;',
        'pub const KNIGHT_BLOCKADE_PENALTY: Score = {value};'
    ),
}

# Piece value pairs need special handling (opening, endgame tuples)
# Format: (opening_param, endgame_param) -> (regex_pattern, replacement_template)
PIECE_VALUE_MAPPINGS = {
    'pawn': {
        'opening_param': 'pawn_value_opening',
        'endgame_param': 'pawn_value_endgame',
        'pattern': r'pub const PAWN_VALUE_PAIR: ScorePair = \(\d+, \d+\);',
        'template': 'pub const PAWN_VALUE_PAIR: ScorePair = ({opening}, {endgame});'
    },
    'knight': {
        'opening_param': 'knight_value_opening',
        'endgame_param': 'knight_value_endgame',
        'pattern': r'pub const KNIGHT_VALUE_PAIR: ScorePair = \(\d+, \d+\);',
        'template': 'pub const KNIGHT_VALUE_PAIR: ScorePair = ({opening}, {endgame});'
    },
    'bishop': {
        'opening_param': 'bishop_value_opening',
        'endgame_param': 'bishop_value_endgame',
        'pattern': r'pub const BISHOP_VALUE_PAIR: ScorePair = \(\d+, \d+\);',
        'template': 'pub const BISHOP_VALUE_PAIR: ScorePair = ({opening}, {endgame});'
    },
    'rook': {
        'opening_param': 'rook_value_opening',
        'endgame_param': 'rook_value_endgame',
        'pattern': r'pub const ROOK_VALUE_PAIR: ScorePair = \(\d+, \d+\);',
        'template': 'pub const ROOK_VALUE_PAIR: ScorePair = ({opening}, {endgame});'
    },
    'queen': {
        'opening_param': 'queen_value_opening',
        'endgame_param': 'queen_value_endgame',
        'pattern': r'pub const QUEEN_VALUE_PAIR: ScorePair = \(\d+, \d+\);',
        'template': 'pub const QUEEN_VALUE_PAIR: ScorePair = ({opening}, {endgame});'
    },
}

# Array parameters need special handling
# These are computed from base + index * per_depth
ARRAY_PARAM_MAPPINGS = {
    'alpha_prune_margins': {
        'pattern': r'pub const ALPHA_PRUNE_MARGINS: \[Score; 8\] = \[[^\]]+\];',
        'template': 'pub const ALPHA_PRUNE_MARGINS: [Score; 8] = [{values}];',
        'size': 8,
        'base_param': 'alpha_prune_margin_base',
        'step_param': 'alpha_prune_margin_per_depth',
    },
    'lmp_move_thresholds': {
        'pattern': r'pub const LMP_MOVE_THRESHOLDS: \[u8; 4\] = \[[^\]]+\];',
        'template': 'pub const LMP_MOVE_THRESHOLDS: [u8; 4] = [{values}];',
        'params': ['0', 'lmp_threshold_depth1', 'lmp_threshold_depth2', 'lmp_threshold_depth3'],
    },
    'passed_pawn_bonus': {
        'pattern': r'pub const VALUE_PASSED_PAWN_BONUS: \[Score; 6\] = \[[^\]]+\];',
        'template': 'pub const VALUE_PASSED_PAWN_BONUS: [Score; 6] = [{values}];',
        'params': ['passed_pawn_bonus_rank2', 'passed_pawn_bonus_rank3', 'passed_pawn_bonus_rank4',
                   'passed_pawn_bonus_rank5', 'passed_pawn_bonus_rank6', 'passed_pawn_bonus_rank7'],
    },
    'connected_passed_pawns': {
        'pattern': r'pub const VALUE_CONNECTED_PASSED_PAWNS: \[Score; 6\] = \[[^\]]+\];',
        'template': 'pub const VALUE_CONNECTED_PASSED_PAWNS: [Score; 6] = [{values}];',
        'params': ['connected_passed_pawn_rank2', 'connected_passed_pawn_rank3', 'connected_passed_pawn_rank4',
                   'connected_passed_pawn_rank5', 'connected_passed_pawn_rank6', 'connected_passed_pawn_rank7'],
    },
}


def get_rusty_rival_path(config: dict = None) -> Path:
    """Get the path to rusty-rival source, using config or default."""
    if config and 'build' in config and 'rusty_rival_path' in config['build']:
        path = config['build']['rusty_rival_path']
    else:
        path = DEFAULT_RUSTY_RIVAL_PATH

    # If relative, resolve from chess-compete directory
    path = Path(path)
    if not path.is_absolute():
        # Get chess-compete root (parent of compete/spsa)
        chess_compete_dir = Path(__file__).parent.parent.parent
        path = chess_compete_dir / path

    return path.resolve()


def read_engine_constants(src_path: Path) -> str:
    """Read the engine_constants.rs file."""
    constants_file = src_path / 'src' / 'engine_constants.rs'
    return constants_file.read_text()


def write_engine_constants(src_path: Path, content: str):
    """Write the engine_constants.rs file."""
    constants_file = src_path / 'src' / 'engine_constants.rs'
    constants_file.write_text(content)


def apply_parameters(content: str, params: dict) -> str:
    """
    Apply parameter values to engine_constants.rs content.

    Args:
        content: Current engine_constants.rs content
        params: Dict of {param_name: value}

    Returns:
        Modified content
    """
    # Apply simple constant mappings
    for param_name, (pattern, template) in PARAM_MAPPINGS.items():
        if param_name in params:
            value = int(round(params[param_name]))
            replacement = template.format(value=value)
            content = re.sub(pattern, replacement, content)

    # Apply ALPHA_PRUNE_MARGINS (computed from base + index * per_depth)
    if 'alpha_prune_margin_base' in params and 'alpha_prune_margin_per_depth' in params:
        base = int(round(params['alpha_prune_margin_base']))
        step = int(round(params['alpha_prune_margin_per_depth']))
        values = [base + i * step for i in range(8)]
        values_str = ', '.join(str(v) for v in values)

        mapping = ARRAY_PARAM_MAPPINGS['alpha_prune_margins']
        replacement = mapping['template'].format(values=values_str)
        content = re.sub(mapping['pattern'], replacement, content)

    # Apply LMP_MOVE_THRESHOLDS
    lmp_params = ['lmp_threshold_depth1', 'lmp_threshold_depth2', 'lmp_threshold_depth3']
    if any(p in params for p in lmp_params):
        values = [0]  # Index 0 is always 0
        for p in lmp_params:
            if p in params:
                values.append(int(round(params[p])))
            else:
                # Default values if not specified
                defaults = {'lmp_threshold_depth1': 8, 'lmp_threshold_depth2': 12, 'lmp_threshold_depth3': 16}
                values.append(defaults[p])
        values_str = ', '.join(str(v) for v in values)

        mapping = ARRAY_PARAM_MAPPINGS['lmp_move_thresholds']
        replacement = mapping['template'].format(values=values_str)
        content = re.sub(mapping['pattern'], replacement, content)

    # Apply PASSED_PAWN_BONUS array
    ppb_params = ['passed_pawn_bonus_rank2', 'passed_pawn_bonus_rank3', 'passed_pawn_bonus_rank4',
                  'passed_pawn_bonus_rank5', 'passed_pawn_bonus_rank6', 'passed_pawn_bonus_rank7']
    if any(p in params for p in ppb_params):
        defaults = [24, 26, 30, 36, 44, 56]
        values = []
        for i, p in enumerate(ppb_params):
            values.append(int(round(params.get(p, defaults[i]))))
        values_str = ', '.join(str(v) for v in values)
        mapping = ARRAY_PARAM_MAPPINGS['passed_pawn_bonus']
        replacement = mapping['template'].format(values=values_str)
        content = re.sub(mapping['pattern'], replacement, content)

    # Apply CONNECTED_PASSED_PAWNS array
    cpp_params = ['connected_passed_pawn_rank2', 'connected_passed_pawn_rank3', 'connected_passed_pawn_rank4',
                  'connected_passed_pawn_rank5', 'connected_passed_pawn_rank6', 'connected_passed_pawn_rank7']
    if any(p in params for p in cpp_params):
        defaults = [12, 18, 28, 42, 60, 80]
        values = []
        for i, p in enumerate(cpp_params):
            values.append(int(round(params.get(p, defaults[i]))))
        values_str = ', '.join(str(v) for v in values)
        mapping = ARRAY_PARAM_MAPPINGS['connected_passed_pawns']
        replacement = mapping['template'].format(values=values_str)
        content = re.sub(mapping['pattern'], replacement, content)

    # Apply piece value pairs (opening, endgame tuples)
    for piece, mapping in PIECE_VALUE_MAPPINGS.items():
        opening_param = mapping['opening_param']
        endgame_param = mapping['endgame_param']
        if opening_param in params or endgame_param in params:
            # Get values, using defaults if one is missing
            defaults = {
                'pawn_value_opening': 100, 'pawn_value_endgame': 200,
                'knight_value_opening': 620, 'knight_value_endgame': 680,
                'bishop_value_opening': 650, 'bishop_value_endgame': 725,
                'rook_value_opening': 1000, 'rook_value_endgame': 1100,
                'queen_value_opening': 2000, 'queen_value_endgame': 2300,
            }
            opening = int(round(params.get(opening_param, defaults[opening_param])))
            endgame = int(round(params.get(endgame_param, defaults[endgame_param])))
            replacement = mapping['template'].format(opening=opening, endgame=endgame)
            content = re.sub(mapping['pattern'], replacement, content)

    return content


def build_engine(src_path: Path, output_path: Path, params: dict = None) -> bool:
    """
    Build the engine with optional parameter modifications.

    Uses a dedicated working copy (rusty-rival-spsa-base) so the original
    source tree is never modified.

    Args:
        src_path: Path to rusty-rival source
        output_path: Path to output the binary (directory)
        params: Optional dict of parameter values to apply

    Returns:
        True if build succeeded
    """
    work_path = src_path.parent / f'{src_path.name}-spsa-base'
    _sync_source_tree(src_path, work_path)

    try:
        _build_in_work_tree(work_path, output_path, params, "base")
        return True
    except RuntimeError as e:
        print(f"Build failed: {e}")
        return False


def _sync_source_tree(src_path: Path, work_path: Path):
    """
    Sync source files to a working copy, preserving target/ for incremental builds.

    Clears old source files but keeps the target/ directory so cargo can do
    incremental compilation.
    """
    if work_path.exists():
        for item in work_path.iterdir():
            if item.name == 'target':
                continue  # Preserve for incremental builds
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    ignore = shutil.ignore_patterns('target', '.git')
    shutil.copytree(src_path, work_path, ignore=ignore, dirs_exist_ok=True)


def _build_in_work_tree(work_path: Path, output_dir: Path, params: dict, label: str) -> Path:
    """
    Apply parameters to a working copy and build.

    Returns the path to the compiled binary.
    """
    # Apply parameters if provided
    if params:
        content = read_engine_constants(work_path)
        content = apply_parameters(content, params)
        write_engine_constants(work_path, content)

    # Build
    env = os.environ.copy()
    if 'RUSTFLAGS' in env:
        del env['RUSTFLAGS']

    result = subprocess.run(
        ['cargo', 'build', '--release'],
        cwd=work_path, env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build {label} engine:\n{result.stderr}")

    # Copy binary to output via temp file + atomic rename to avoid
    # "Text file busy" errors when the old binary is still running
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_name = 'rusty-rival.exe' if os.name == 'nt' else 'rusty-rival'
    src_binary = work_path / 'target' / 'release' / binary_name
    dst_binary = output_dir / binary_name
    tmp_binary = output_dir / f'.{binary_name}.tmp'
    shutil.copy2(src_binary, tmp_binary)
    os.replace(tmp_binary, dst_binary)

    return dst_binary


def build_spsa_engines(src_path: Path, output_base: Path,
                       plus_params: dict, minus_params: dict,
                       plus_name: str = 'spsa-plus', minus_name: str = 'spsa-minus') -> tuple[Path, Path]:
    """
    Build both plus and minus perturbed engines for SPSA iteration.

    Uses two working copies of the source tree so both engines can be built
    in parallel. Each copy preserves its target/ directory for incremental
    compilation.

    Args:
        src_path: Path to rusty-rival source
        output_base: Base path for engine output directories
        plus_params: Parameter values for plus engine
        minus_params: Parameter values for minus engine
        plus_name: Name for plus engine directory
        minus_name: Name for minus engine directory

    Returns:
        (plus_engine_path, minus_engine_path) - full paths to engine binaries
    """
    from concurrent.futures import ThreadPoolExecutor

    plus_dir = output_base / plus_name
    minus_dir = output_base / minus_name

    # Create/sync two working copies of the source tree
    plus_work = src_path.parent / f'{src_path.name}-spsa-plus'
    minus_work = src_path.parent / f'{src_path.name}-spsa-minus'

    print(f"  Syncing source trees...")
    _sync_source_tree(src_path, plus_work)
    _sync_source_tree(src_path, minus_work)

    print(f"  Building plus and minus engines in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        plus_future = executor.submit(_build_in_work_tree, plus_work, plus_dir, plus_params, "plus")
        minus_future = executor.submit(_build_in_work_tree, minus_work, minus_dir, minus_params, "minus")

        plus_path = plus_future.result()
        minus_path = minus_future.result()

    return plus_path, minus_path
