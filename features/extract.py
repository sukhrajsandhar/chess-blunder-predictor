import chess
import chess.pgn
import re
import json
import os
import glob

def clk_to_seconds(clk_str):
    parts = clk_str.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s

def get_game_phase(board, move_number):
    total_material = sum(
        len(board.pieces(pt, c)) * val
        for pt, val in [(chess.QUEEN,9),(chess.ROOK,5),(chess.BISHOP,3),(chess.KNIGHT,3)]
        for c in [chess.WHITE, chess.BLACK]
    )
    if move_number <= 10:
        return 0
    elif total_material <= 20:
        return 2
    else:
        return 1

def board_to_tensor(board):
    piece_map = {
        (chess.PAWN,   chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK,   chess.WHITE): 3,
        (chess.QUEEN,  chess.WHITE): 4,
        (chess.KING,   chess.WHITE): 5,
        (chess.PAWN,   chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK,   chess.BLACK): 9,
        (chess.QUEEN,  chess.BLACK): 10,
        (chess.KING,   chess.BLACK): 11,
    }
    tensor = [[[0]*8 for _ in range(8)] for _ in range(12)]
    for square, piece in board.piece_map().items():
        rank = square // 8
        file = square % 8
        channel = piece_map[(piece.piece_type, piece.color)]
        tensor[channel][rank][file] = 1
    return tensor

def parse_game(game):
    board = game.board()
    moves = []
    prev_clk_white = None
    prev_clk_black = None
    time_spent_history = []
    eval_history = []

    try:
        white_elo = int(game.headers.get("WhiteElo", 1500))
    except:
        white_elo = 1500
    try:
        black_elo = int(game.headers.get("BlackElo", 1500))
    except:
        black_elo = 1500

    try:
        tc = game.headers.get("TimeControl", "600+0")
        base_time = int(tc.split("+")[0]) if "+" in tc else int(tc)
    except:
        base_time = 600

    node = game
    move_number = 0

    while node.variations:
        next_node = node.variations[0]
        comment = next_node.comment
        move_number += 1

        clk_match = re.search(r'\[%clk (\d+:\d+:\d+)\]', comment)
        eval_match = re.search(r'\[%eval (-?\d+\.?\d*)\]', comment)

        clk_seconds = clk_to_seconds(clk_match.group(1)) if clk_match else None
        eval_score = float(eval_match.group(1)) if eval_match else None

        is_white = (move_number % 2 == 1)
        current_player = "white" if is_white else "black"
        player_elo = white_elo if is_white else black_elo

        if is_white:
            time_spent = (prev_clk_white - clk_seconds) if (prev_clk_white is not None and clk_seconds is not None) else 0
            prev_clk_white = clk_seconds
        else:
            time_spent = (prev_clk_black - clk_seconds) if (prev_clk_black is not None and clk_seconds is not None) else 0
            prev_clk_black = clk_seconds

        time_spent = max(0, time_spent)
        time_spent_history.append(time_spent)

        avg_time = sum(time_spent_history) / len(time_spent_history) if time_spent_history else 1
        time_ratio = time_spent / avg_time if avg_time > 0 else 1.0
        time_pressure = 1 if (clk_seconds is not None and clk_seconds < 300) else 0

        eval_history.append(eval_score if eval_score is not None else 0)
        if len(eval_history) >= 3:
            eval_trend = eval_history[-1] - eval_history[-3]
        else:
            eval_trend = 0.0

        legal_moves = board.legal_moves.count()
        game_phase = get_game_phase(board, move_number)

        white_material = sum(
            len(board.pieces(pt, chess.WHITE)) * val
            for pt, val in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]
        )
        black_material = sum(
            len(board.pieces(pt, chess.BLACK)) * val
            for pt, val in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]
        )
        material_balance = white_material - black_material

        board_tensor = board_to_tensor(board)

        moves.append({
            "move_number": move_number,
            "player": current_player,
            "move": next_node.move.uci(),
            "clk_seconds": clk_seconds,
            "time_spent": time_spent,
            "time_ratio": round(time_ratio, 3),
            "time_pressure": time_pressure,
            "eval": eval_score,
            "eval_trend": round(eval_trend, 3),
            "legal_moves": legal_moves,
            "material_balance": material_balance,
            "player_elo": player_elo,
            "base_time": base_time,
            "game_phase": game_phase,
            "board": board_tensor,
        })

        board.push(next_node.move)
        node = next_node

    for i in range(1, len(moves)):
        prev_eval = moves[i-1]["eval"]
        curr_eval = moves[i]["eval"]
        if prev_eval is not None and curr_eval is not None:
            eval_change = curr_eval - prev_eval
            if moves[i]["player"] == "white":
                moves[i]["is_blunder"] = 1 if eval_change <= -1.5 else 0
            else:
                moves[i]["is_blunder"] = 1 if eval_change >= 1.5 else 0
        else:
            moves[i]["is_blunder"] = 0

    moves[0]["is_blunder"] = 0
    return moves

if __name__ == "__main__":
    pgn_files = glob.glob("data/raw/*.pgn")
    output_path = "data/all_features.json"
    MAX_GAMES = 50000

    all_moves = []
    game_count = 0
    skipped = 0
    done = False

    for pgn_file in pgn_files:
        if done:
            break
        print(f"Processing {pgn_file}...")
        with open(pgn_file, encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                try:
                    moves = parse_game(game)
                    evals_present = sum(1 for m in moves if m["eval"] is not None)
                    if len(moves) >= 20 and evals_present >= 10:
                        all_moves.extend(moves)
                        game_count += 1
                        if game_count % 5000 == 0:
                            print(f"  {game_count} games, {len(all_moves)} moves...")
                        if game_count >= MAX_GAMES:
                            done = True
                            break
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
                    continue

        print(f"  Finished {pgn_file}: {game_count} games so far")

    blunders = sum(1 for m in all_moves if m.get("is_blunder") == 1)
    print(f"\nDone!")
    print(f"Games: {game_count} | Skipped: {skipped}")
    print(f"Total moves: {len(all_moves)}")
    print(f"Blunders: {blunders} ({100*blunders/len(all_moves):.1f}%)")

    with open(output_path, "w") as f:
        json.dump(all_moves, f)
    print(f"Saved to {output_path}")