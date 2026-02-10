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

    # Copy binary to output
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_name = 'rusty-rival.exe' if os.name == 'nt' else 'rusty-rival'
    src_binary = work_path / 'target' / 'release' / binary_name
    dst_binary = output_dir / binary_name
    shutil.copy2(src_binary, dst_binary)

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


if __name__ == '__main__':
    # Test: build with current params
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    spsa_dir = Path(__file__).parent

    # Load config to get rusty-rival path
    with open(spsa_dir / 'config.toml', 'rb') as f:
        config = tomllib.load(f)

    src_path = get_rusty_rival_path(config)

    # Load params
    with open(spsa_dir / 'params.toml', 'rb') as f:
        params_config = tomllib.load(f)

    # Extract current values
    params = {name: cfg['value'] for name, cfg in params_config.items()}

    print(f"Rusty-rival path: {src_path}")
    print(f"Parameters: {params}")
    print(f"Building engine with current parameters...")

    output_path = spsa_dir / 'test-build'
    if build_engine(src_path, output_path, params):
        print(f"Success! Binary at: {output_path}")
    else:
        print("Build failed!")
