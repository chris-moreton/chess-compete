"""
Chess opening book and EPD position loading.
"""

from pathlib import Path


def load_epd_positions(epd_file: Path) -> list[tuple[str, str]]:
    """
    Load positions from an EPD file.
    Returns list of (fen, position_id) tuples.

    EPD format: FEN [operations]
    Example: 8/8/p2p3p/3k2p1/PP6/3K1P1P/8/8 b - - bm Kc6; id "E_E_T 001";

    We extract the FEN (first 4 fields) and the id if present.
    If no id, we use the position number.
    """
    positions = []
    with open(epd_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # EPD has 4 FEN fields (board, side, castling, en passant)
            # followed by optional operations like bm, id, etc.
            parts = line.split()
            if len(parts) < 4:
                continue

            # Construct FEN with default halfmove/fullmove counts
            fen = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} 0 1"

            # Try to extract id from the line
            pos_id = None
            if 'id "' in line:
                try:
                    start = line.index('id "') + 4
                    end = line.index('"', start)
                    pos_id = line[start:end]
                except ValueError:
                    pass

            if pos_id is None:
                pos_id = f"Position {line_num}"

            positions.append((fen, pos_id))

    return positions


# Opening positions (FEN) - balanced positions after 4-8 moves from various openings
# Each position will be played twice (once with each engine as white)
# 250 positions = 500 unique games maximum
OPENING_BOOK = [
    # ============ SICILIAN DEFENSE (25 variations) ============
    ("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 1 5", "Sicilian Open"),
    ("r1bqkb1r/pp1ppppp/2n2n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 4 5", "Sicilian Nc6"),
    ("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP1BPPP/RNBQK2R b KQkq - 1 5", "Sicilian Najdorf Be2"),
    ("r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6", "Sicilian Scheveningen"),
    ("r1bqkb1r/pp3ppp/2nppn2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 7", "Sicilian Dragon"),
    ("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6", "Sicilian Najdorf"),
    ("r1bqkb1r/pp2pppp/2np1n2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R b KQkq - 5 6", "Sicilian Richter-Rauzer"),
    ("r1b1kb1r/pp2pppp/1qnp1n2/8/3NP3/2N5/PPP1BPPP/R1BQK2R w KQkq - 4 7", "Sicilian Sozin"),
    ("r1bqk2r/pp2bppp/2nppn2/8/3NP3/2N1B3/PPP1BPPP/R2QK2R b KQkq - 2 8", "Sicilian English Attack"),
    ("r1bqkb1r/5ppp/p1np1n2/1p2p3/4P3/N1N5/PPP1BPPP/R1BQK2R w KQkq - 0 9", "Sicilian Sveshnikov"),
    ("r1bqk2r/pp2bppp/2nppn2/8/4P3/1NN5/PPP1BPPP/R1BQK2R b KQkq - 1 8", "Sicilian Maroczy Bind"),
    ("r1bqkb1r/pp3ppp/2nppn2/8/3NP3/2N5/PPP1BPPP/R1BQK2R w KQkq - 0 7", "Sicilian Classical"),
    ("r1bqkb1r/1p2pppp/p1np1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 7", "Sicilian Najdorf Be3"),
    ("r1bqk2r/pp2bppp/2nppn2/8/3NP3/2N1BP2/PPP3PP/R2QKB1R b KQkq - 0 8", "Sicilian Dragon Yugoslav"),
    ("r1b1kb1r/pp3ppp/1qnppn2/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 2 8", "Sicilian Taimanov"),
    ("rnbqkb1r/pp3ppp/3ppn2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6", "Sicilian Kan"),
    ("r1bqkb1r/pp2pppp/2np1n2/8/2BNP3/2N5/PPP2PPP/R1BQK2R b KQkq - 5 6", "Sicilian Bc4"),
    ("rnbqkb1r/pp2pppp/3p1n2/8/2BNP3/8/PPP2PPP/RNBQK2R b KQkq - 1 5", "Sicilian Open Bc4"),
    ("r1bqk2r/pp1nbppp/2npp3/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 2 8", "Sicilian Paulsen"),
    ("r1bqkb1r/pp3ppp/2nppn2/1B6/3NP3/2N5/PPP2PPP/R1BQK2R b KQkq - 1 7", "Sicilian Bb5+"),
    ("rnbqkb1r/pp2pppp/3p1n2/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4", "Sicilian c5"),
    ("r1bqkb1r/pp3ppp/2nppn2/8/3NP3/2N2P2/PPP3PP/R1BQKB1R b KQkq - 0 7", "Sicilian f3"),
    ("r1bqkb1r/pp2pp1p/2np1np1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 7", "Sicilian Accelerated Dragon"),
    ("rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6", "Sicilian Dragon Setup"),
    ("r1bqk2r/pp2bppp/2np1n2/4p3/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 0 8", "Sicilian Boleslavsky"),

    # ============ ITALIAN GAME / GIUOCO PIANO (15 variations) ============
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian Game"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Two Knights"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5", "Italian Giuoco Piano"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq - 0 5", "Italian d4"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 2 5", "Italian Nc3"),
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 4 7", "Italian Quiet"),
    ("r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 2 5", "Italian Evans Gambit"),
    ("r1bqkb1r/pppp1Npp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4", "Italian Fried Liver"),
    ("r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 6", "Italian d6"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 5", "Italian O-O"),
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP1QPPP/RNB2RK1 b - - 6 7", "Italian Qe2"),
    ("r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQK2R b KQkq - 0 6", "Italian Main Line"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 5", "Four Knights Italian"),
    ("r1bqk2r/pppp1ppp/2n2n2/4p3/1bBPP3/2N2N2/PPP2PPP/R1BQK2R b KQkq - 0 5", "Italian Scotch Gambit"),
    ("r1bq1rk1/ppppbppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQR1K1 w - - 6 7", "Italian Re1"),

    # ============ RUY LOPEZ (20 variations) ============
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "Ruy Lopez"),
    ("r1bqkb1r/1ppp1ppp/p1n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5", "Ruy Lopez Morphy"),
    ("r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 5", "Ruy Lopez Closed"),
    ("r1bqkb1r/2pp1ppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQK2R w KQkq - 0 6", "Ruy Lopez Marshall"),
    ("r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQ1RK1 w kq - 2 7", "Ruy Lopez Be7"),
    ("r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 2 5", "Ruy Lopez O-O"),
    ("r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 3 6", "Ruy Lopez Main Line"),
    ("r1bqkb1r/2pp1ppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQ1RK1 b kq - 1 6", "Ruy Lopez Archangel"),
    ("r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQ1RK1 b kq - 0 7", "Ruy Lopez c3"),
    ("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 b - - 2 8", "Ruy Lopez Re1"),
    ("r1bqkb1r/1ppp1ppp/p1B2n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 5", "Ruy Lopez Exchange"),
    ("r1bqk2r/2ppbppp/p1n2n2/1p2p3/P3P3/1B3N2/1PPP1PPP/RNBQ1RK1 b kq - 0 7", "Ruy Lopez a4"),
    ("r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 6", "Ruy Lopez Nc3"),
    ("r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B1P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 7", "Ruy Lopez d3"),
    ("r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N1P/PPPP1PP1/RNBQ1RK1 b kq - 0 7", "Ruy Lopez h3"),
    ("r2qkb1r/1bpppppp/p1n2n2/1p6/4P3/1B3N2/PPPP1PPP/RNBQ1RK1 w kq - 2 7", "Ruy Lopez Berlin"),
    ("r1bqk2r/2ppbppp/p1n2n2/1p2N3/4P3/1B6/PPPP1PPP/RNBQ1RK1 b kq - 0 7", "Ruy Lopez Ne5"),
    ("r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9", "Ruy Lopez d6 Closed"),
    ("r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPPQPPP/RNB2RK1 b kq - 3 7", "Ruy Lopez Qe2"),
    ("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 b - - 2 8", "Ruy Lopez Normal"),

    # ============ QUEEN'S GAMBIT (20 variations) ============
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4", "QGD"),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3", "QG Accepted"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4", "QGD Orthodox"),
    ("rnbqkb1r/p1p2ppp/1p2pn2/3p4/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 0 5", "QGD Tartakower"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4", "QGD Two Knights"),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 2 3", "QGD Nc3"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 4 5", "QGD Ragozin"),
    ("rnbqkb1r/p1p2ppp/1p2pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5", "QGD Tartakower Alt"),
    ("rn1qkb1r/ppp1pppp/4bn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 4", "QGD Bf5"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR b KQkq - 3 4", "QGD Bg5"),
    ("rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5", "QGD Tarrasch"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 4", "QGD Catalan"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 2 5", "QGD Catalan Be7"),
    ("rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 5", "QGD Queen's Indian"),
    ("rnbqkb1r/pp3ppp/4pn2/2pP4/3P4/2N5/PP2PPPP/R1BQKBNR b KQkq - 0 5", "QGD Semi-Tarrasch"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/4PN2/PP3PPP/RNBQKB1R b KQkq - 0 4", "QGD e3"),
    ("rnbqkb1r/ppp2p1p/4pnp1/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5", "QGD Schlechter"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 4 5", "QGD Exchange"),
    ("rn1qkb1r/ppp1pppp/4bn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 4 4", "QGD Early Bf5"),
    ("rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq c6 0 5", "QGD Tarrasch Main"),

    # ============ KING'S INDIAN DEFENSE (15 variations) ============
    ("rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5", "King's Indian"),
    ("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 3 6", "KID Classical"),
    ("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 2 6", "KID Samisch"),
    ("rnbq1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 7", "KID Mar del Plata"),
    ("rnbq1rk1/ppp1ppbp/3p1np1/8/2PP4/2N2NP1/PP2PP1P/R1BQKB1R b KQ - 0 6", "KID Fianchetto"),
    ("rnbq1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N5/PP2BPPP/R1BQK1NR w KQ - 0 7", "KID Four Pawns"),
    ("rnbq1rk1/pp2ppbp/2pp1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 7", "KID Petrosian"),
    ("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N1BN2/PP3PPP/R2QKB1R b KQ - 4 6", "KID Be3"),
    ("rnbq1rk1/ppp2pbp/3p1np1/8/2PPp3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 7", "KID e4e5"),
    ("rnbqr1k1/ppp2pbp/3p1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 2 8", "KID Re8"),
    ("r1bq1rk1/ppp1ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 2 7", "KID Nc6"),
    ("rnbq1rk1/ppp2pbp/3p1np1/4p3/2PP4/2N2NP1/PP2PPBP/R1BQK2R b KQ - 0 7", "KID Fianchetto Main"),
    ("rnbq1rk1/pp2ppbp/2pp1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 0 7", "KID Averbakh"),
    ("rnbq1rk1/ppp1ppbp/3p1np1/6B1/2PPP3/2N5/PP3PPP/R2QKBNR b KQ - 3 6", "KID Bg5"),
    ("rnbq1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N2P2/PP4PP/R1BQKBNR b KQ - 0 7", "KID Samisch f3"),

    # ============ FRENCH DEFENSE (15 variations) ============
    ("rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3", "French Advance"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4", "French Nc3"),
    ("rnbqkb1r/ppp2ppp/4pn2/3pP3/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 4", "French Steinitz"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 4 5", "French Classical"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 4", "French Exchange"),
    ("rnbqk2r/ppp1bppp/4pn2/3pP3/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq - 2 5", "French Steinitz Main"),
    ("rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 1 3", "French Advance Nf3"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 4 5", "French Classical Be7"),
    ("rnbqkb1r/ppp2ppp/4pn2/3pP3/3P2P1/8/PPP2P1P/RNBQKBNR b KQkq - 0 4", "French g4"),
    ("rnbqk2r/ppp2ppp/4pn2/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 3 5", "French Winawer"),
    ("rnbqk1nr/ppp2ppp/4p3/3pP3/1b1P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4", "French Winawer Main"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/3PP3/2N2P2/PPP3PP/R1BQKBNR b KQkq - 0 5", "French Rubinstein"),
    ("rnbqk2r/ppp1bppp/4pn2/3pP3/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 3 5", "French Classical Nf3"),
    ("rnbqkb1r/ppp2ppp/4pn2/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5", "French Rubinstein exd5"),
    ("rnbqk2r/ppp1bppp/4p3/3pP2n/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 3 5", "French Alekhine-Chatard"),

    # ============ CARO-KANN (12 variations) ============
    ("rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3", "Caro-Kann Advance"),
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4", "Caro-Kann Classical"),
    ("rnbqkb1r/pp2pppp/5n2/2pp4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4", "Caro-Kann Panov"),
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 4", "Caro-Kann Main"),
    ("rn1qkbnr/pp2pppp/2p5/3pPb2/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 1 4", "Caro-Kann Advance Bf5"),
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 3 4", "Caro-Kann Two Knights"),
    ("rn1qkbnr/pp2pppp/2p5/3pPb2/3P2P1/8/PPP2P1P/RNBQKBNR b KQkq - 0 4", "Caro-Kann g4"),
    ("rnbqkb1r/pp2pppp/2p2n2/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5", "Caro-Kann Nxe4"),
    ("rn1qkbnr/pp2pppp/2p5/3pPb2/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 4", "Caro-Kann Short"),
    ("rnbqkb1r/pp2pppp/5n2/2pP4/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 4", "Caro-Kann Panov Main"),
    ("rnbqkb1r/pp3ppp/2p2n2/3pp3/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5", "Caro-Kann e5"),
    ("rn1qkb1r/pp2pppp/2p2n2/3p4/3PP1b1/2N5/PPP2PPP/R1BQKBNR w KQkq - 3 5", "Caro-Kann Bg4"),

    # ============ ENGLISH OPENING (12 variations) ============
    ("rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 2", "English vs e5"),
    ("rnbqkb1r/pppppppp/5n2/8/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 2 2", "English Nf3"),
    ("rnbqkbnr/pp1ppppp/8/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 1 2", "English Symmetrical"),
    ("rnbqkbnr/pp1ppppp/8/2p5/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq - 1 2", "English Nc3"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3", "English Nc6"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3", "English Reversed Sicilian"),
    ("rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR w KQkq d6 0 2", "English vs d5"),
    ("rnbqkbnr/pp1ppppp/8/2p5/2PP4/8/PP2PPPP/RNBQKBNR b KQkq d3 0 2", "English d4"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 2 3", "English Bremen"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2P5/5NP1/PP1PPP1P/RNBQKB1R b KQkq - 0 3", "English King's"),
    ("rnbqkbnr/p1pppppp/1p6/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2", "English vs b6"),
    ("rnbqkb1r/pp1ppppp/5n2/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 2 3", "English Hedgehog"),

    # ============ SLAV DEFENSE (10 variations) ============
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4", "Slav Defense"),
    ("rnbqkb1r/p3pppp/2p2n2/1p1p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 5", "Slav Chebanenko"),
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 3 4", "Slav Nc3"),
    ("rn1qkb1r/pp2pppp/2p2n2/3p1b2/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5", "Slav Bf5"),
    ("rnbqkb1r/pp2pppp/2p2n2/8/2pP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 5", "Slav Exchange"),
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4", "Slav Two Knights"),
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/4PN2/PP3PPP/RNBQKB1R b KQkq - 0 4", "Slav e3"),
    ("rn1qkb1r/pp2pppp/2p2n2/3p4/2PP2b1/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5", "Slav Bg4"),
    ("rnbqkb1r/1p2pppp/p1p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 5", "Slav a6"),
    ("rnbqkb1r/pp3ppp/2p2n2/3pp3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq e6 0 5", "Slav Semi-Slav"),

    # ============ NIMZO-INDIAN (12 variations) ============
    ("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4", "Nimzo-Indian"),
    ("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4", "Nimzo-Indian Classical"),
    ("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N1P3/PP3PPP/R1BQKBNR b KQkq - 0 4", "Nimzo-Indian Rubinstein"),
    ("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PPQ1PPPP/R1B1KBNR b KQkq - 3 4", "Nimzo-Indian Qc2"),
    ("rnbqk2r/p1pp1ppp/1p2pn2/8/1bPP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5", "Nimzo-Indian b6"),
    ("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/2N1P3/PP3PPP/R1BQKBNR w KQ - 1 5", "Nimzo-Indian O-O"),
    ("rnbqk2r/pppp1ppp/4pn2/6B1/1bPP4/2N5/PP2PPPP/R2QKBNR b KQkq - 3 4", "Nimzo-Indian Bg5"),
    ("rnbq1rk1/p1pp1ppp/1p2pn2/8/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 6", "Nimzo-Indian Fischer"),
    ("rnbqk2r/pppp1ppp/4pn2/8/2PP4/P1b5/1P2PPPP/R1BQKBNR w KQkq - 0 5", "Nimzo-Indian Samisch"),
    ("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/2N2N2/PP2PPPP/R1BQKB1R w KQ - 4 5", "Nimzo-Indian Main"),
    ("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N1P3/PP1B1PPP/R2QKBNR b KQkq - 2 5", "Nimzo-Indian Bd2"),
    ("rnbq1rk1/ppp2ppp/4pn2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 6", "Nimzo-Indian d5"),

    # ============ SCOTCH GAME (8 variations) ============
    ("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3", "Scotch Game"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4N3/3PP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4", "Scotch Four Knights"),
    ("r1bqkbnr/pppp1ppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4", "Scotch Main"),
    ("r1bqkb1r/pppp1ppp/2n2n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5", "Scotch Nf6"),
    ("r1bqkbnr/pppp1ppp/8/4n3/3PP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 4", "Scotch Nxe4"),
    ("r1bqk1nr/pppp1ppp/2n5/2b5/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5", "Scotch Classical"),
    ("r1bqkbnr/pppp1ppp/2n5/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 4", "Scotch Nc3"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b5/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 4 5", "Scotch Four Knights Bc5"),

    # ============ PETROV DEFENSE (8 variations) ============
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Petrov Defense"),
    ("rnbqkb1r/pppp1ppp/8/4p3/4n3/5N2/PPPPQPPP/RNB1KB1R b KQkq - 3 4", "Petrov Classical"),
    ("rnbqkb1r/pppp1ppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 3", "Petrov Nxe5"),
    ("rnbqkb1r/ppp2ppp/3p1n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 4", "Petrov d6"),
    ("rnbqkb1r/pppp1ppp/8/4p3/3Pn3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 4", "Petrov d4"),
    ("rnbqkb1r/ppp2ppp/3p1n2/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 4", "Petrov Steinitz"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 3 3", "Petrov Three Knights"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4", "Petrov Four Knights"),

    # ============ PIRC DEFENSE (8 variations) ============
    ("rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 3", "Pirc Defense"),
    ("rnbqkb1r/ppp1pp1p/3p1np1/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4", "Pirc Austrian"),
    ("rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 3 3", "Pirc Nc3"),
    ("rnbqkb1r/ppp1pp1p/3p1np1/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 1 4", "Pirc Classical"),
    ("rnbqkb1r/ppp1pp1p/3p1np1/8/3PPP2/2N5/PPP3PP/R1BQKBNR b KQkq - 0 4", "Pirc Austrian f4"),
    ("rnbqk2r/ppp1ppbp/3p1np1/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 5", "Pirc Classical Bg7"),
    ("rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 3 3", "Pirc Nf3"),
    ("rnbqk2r/ppp1ppbp/3p1np1/8/3PP1P1/2N5/PPP2P1P/R1BQKBNR b KQkq - 0 5", "Pirc g4"),

    # ============ SCANDINAVIAN (6 variations) ============
    ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "Scandinavian"),
    ("rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 3", "Scandinavian Qxd5"),
    ("rnbqkbnr/ppp1pppp/8/8/4p3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 3", "Scandinavian Nxd5"),
    ("rnb1kbnr/ppp1pppp/8/3q4/8/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 3", "Scandinavian Nf3"),
    ("rnb1kbnr/ppp1pppp/3q4/8/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4", "Scandinavian Qd6"),
    ("rn1qkbnr/ppp1pppp/8/3p1b2/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4", "Scandinavian Bf5"),

    # ============ LONDON SYSTEM (8 variations) ============
    ("rnbqkb1r/ppp1pppp/3p1n2/8/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3", "London System"),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3", "London vs d5"),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/4P3/PPP2PPP/RN1QKBNR b KQkq - 0 3", "London e3"),
    ("rnbqk2r/ppp1ppbp/3p1np1/8/3P1B2/5N2/PPP1PPPP/RN1QKB1R w KQkq - 0 5", "London vs KID"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R w KQkq - 0 4", "London vs e6"),
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R w KQkq - 0 4", "London vs Slav"),
    ("rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3", "London Torre"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/3P1B2/4PN2/PPP2PPP/RN1QKB1R w KQkq - 2 5", "London Classical"),

    # ============ CATALAN (8 variations) ============
    ("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3", "Catalan"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/6P1/PP2PP1P/RNBQKBNR w KQkq - 0 4", "Catalan Open"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/6P1/PP2PPBP/RNBQK1NR w KQkq - 2 5", "Catalan Be7"),
    ("rnbqkb1r/ppp2ppp/4pn2/8/2pP4/6P1/PP2PPBP/RNBQK1NR w KQkq - 0 5", "Catalan dxc4"),
    ("rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQK2R b KQkq - 3 5", "Catalan Main"),
    ("rnbqkb1r/p1p2ppp/1p2pn2/3p4/2PP4/6P1/PP2PPBP/RNBQK1NR w KQkq - 0 5", "Catalan b6"),
    ("rnbqk2r/ppp2ppp/4pn2/3p4/1bPP4/6P1/PP2PPBP/RNBQK1NR w KQkq - 2 5", "Catalan Bb4"),
    ("rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQK2R w KQ - 4 6", "Catalan Closed"),

    # ============ GRUNFELD (8 variations) ============
    ("rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq d6 0 4", "Grunfeld"),
    ("rnbqkb1r/ppp1pp1p/6p1/3n4/3P4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 5", "Grunfeld Exchange"),
    ("rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/1QN5/PP2PPPP/R1B1KBNR b KQkq - 3 4", "Grunfeld Russian"),
    ("rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4", "Grunfeld Two Knights"),
    ("rnbqk2r/ppp1ppbp/5np1/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 5", "Grunfeld Bg7"),
    ("rnbqkb1r/ppp1pp1p/6p1/3n4/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq - 0 5", "Grunfeld Exchange e4"),
    ("rnbqk2r/ppp1ppbp/5np1/8/2pP4/2N2NP1/PP2PP1P/R1BQKB1R w KQkq - 0 6", "Grunfeld Fianchetto"),
    ("rnbqkb1r/ppp1pp1p/5np1/3p4/2PP1B2/2N5/PP2PPPP/R2QKBNR b KQkq - 3 4", "Grunfeld Bf4"),

    # ============ VIENNA GAME (6 variations) ============
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3", "Vienna Game"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4PP2/2N5/PPPP2PP/R1BQKBNR b KQkq - 0 3", "Vienna Gambit"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/2N5/PPPP1PPP/R1BQK1NR b KQkq - 3 3", "Vienna Bc4"),
    ("rnbqkb1r/pppp1ppp/8/4p3/4Pn2/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 4", "Vienna Nxe4"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4", "Vienna Four Knights"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N3P1/PPPP1P1P/R1BQKBNR b KQkq - 0 3", "Vienna g3"),

    # ============ KING'S GAMBIT (6 variations) ============
    ("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 2", "King's Gambit"),
    ("rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3", "King's Gambit Accepted"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2", "King's Gambit Main"),
    ("rnbqkbnr/pppp1ppp/8/8/4Pp2/5N2/PPPP2PP/RNBQKB1R b KQkq - 1 3", "King's Gambit Nf3"),
    ("rnbqkbnr/ppp2ppp/8/3pp3/4PP2/8/PPPP2PP/RNBQKBNR w KQkq d6 0 3", "King's Gambit Declined"),
    ("rnbqk1nr/pppp1ppp/8/2b1p3/4PP2/8/PPPP2PP/RNBQKBNR w KQkq - 1 3", "King's Gambit Classical"),

    # ============ MODERN DEFENSE (6 variations) ============
    ("rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Modern Defense"),
    ("rnbqkbnr/pppppp1p/6p1/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2", "Modern d4"),
    ("rnbqk1nr/ppppppbp/6p1/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 2 3", "Modern Averbakh"),
    ("rnbqk1nr/ppppppbp/6p1/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3", "Modern Bg7"),
    ("rnbqk1nr/ppp1ppbp/3p2p1/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4", "Modern Pirc-like"),
    ("rnbqk1nr/ppppppbp/6p1/8/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3", "Modern Nf3"),

    # ============ ALEKHINE'S DEFENSE (6 variations) ============
    ("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", "Alekhine's Defense"),
    ("rnbqkb1r/pppppppp/8/4n3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3", "Alekhine's e5"),
    ("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3", "Alekhine's Four Pawns"),
    ("rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 4", "Alekhine's Modern"),
    ("rnbqkb1r/ppp1pppp/3p1n2/4P3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 4", "Alekhine's Exchange"),
    ("rnbqkb1r/pppppppp/5n2/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2", "Alekhine's Chase"),

    # ============ BENONI (8 variations) ============
    ("rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P5/8/PP2PPPP/RNBQKBNR w KQkq - 0 4", "Benoni"),
    ("rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P5/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 4", "Modern Benoni"),
    ("rnbqkb1r/pp3ppp/3ppn2/2pP4/2P5/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 5", "Benoni Main"),
    ("rnbqkb1r/pp1p1ppp/4pn2/8/2pP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 5", "Benoni Benko"),
    ("rnbqk2r/pp1p1ppp/4pn2/2pP4/2P5/2Nb4/PP2PPPP/R1BQKBNR w KQkq - 2 5", "Benoni Bb4"),
    ("rnbqkb1r/pp3ppp/4pn2/2pP4/8/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 5", "Benoni d5"),
    ("rnbq1rk1/pp1p1ppp/4pn2/2pP4/2P5/2N2N2/PP2PPPP/R1BQKB1R w KQ - 2 6", "Benoni O-O"),
    ("rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P1P3/8/PP3PPP/RNBQKBNR b KQkq - 0 4", "Benoni e4"),

    # ============ DUTCH DEFENSE (6 variations) ============
    ("rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq f6 0 2", "Dutch Defense"),
    ("rnbqkbnr/ppppp1pp/8/5p2/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2", "Dutch c4"),
    ("rnbqkb1r/ppppp1pp/5n2/5p2/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3", "Dutch Leningrad"),
    ("rnbqkb1r/pppp2pp/4pn2/5p2/2PP4/6P1/PP2PP1P/RNBQKBNR w KQkq - 0 4", "Dutch Stonewall"),
    ("rnbqkb1r/ppppp1pp/5n2/5p2/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 2 3", "Dutch Classical"),
    ("rnbqk2r/ppppp1bp/5np1/5p2/2PP4/6P1/PP2PPBP/RNBQK1NR w KQkq - 2 5", "Dutch Leningrad Main"),

    # ============ MISCELLANEOUS OPENINGS (10 variations) ============
    ("rnbqkbnr/ppp1pppp/8/3p4/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq - 0 2", "Reti Opening"),
    ("rnbqkbnr/pppp1ppp/8/4p3/6P1/8/PPPPPP1P/RNBQKBNR w KQkq e6 0 2", "Grob Attack"),
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "Queen's Pawn"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 3", "English e5"),
    ("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "French 1...e6"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3", "English e5 Nc6"),
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "Sicilian 1...c5"),
    ("rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Sicilian Nf6"),
    ("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Caro-Kann 1...c6"),
    ("rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Pirc 1...d6"),
]
