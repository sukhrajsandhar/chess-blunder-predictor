import chess.pgn
import re
import json
import os

def clk_to_seconds(clk_str):
    parts = clk_str.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s

def parse_game(game):
    board = game.board()
    moves = []
    prev_clk_white = None
    prev_clk_black = None
    time_spent_history = []
    eval_history = []

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

        white_material = sum(
            len(board.pieces(pt, chess.WHITE)) * val
            for pt, val in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]
        )
        black_material = sum(
            len(board.pieces(pt, chess.BLACK)) * val
            for pt, val in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]
        )
        material_balance = white_material - black_material

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
        })

        board.push(next_node.move)
        node = next_node

    # Label blunders
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
    pgn_path = "data/raw/games.pgn"
    output_path = "data/all_features.json"

    all_moves = []
    game_count = 0
    skipped = 0

    with open(pgn_path, encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            try:
                moves = parse_game(game)
                # Only keep games with at least 20 moves and some evals
                evals_present = sum(1 for m in moves if m["eval"] is not None)
                if len(moves) >= 20 and evals_present >= 10:
                    all_moves.extend(moves)
                    game_count += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                continue

            if game_count % 50 == 0:
                print(f"Processed {game_count} games, {len(all_moves)} total moves...")

    blunders = sum(1 for m in all_moves if m.get("is_blunder") == 1)
    non_blunders = sum(1 for m in all_moves if m.get("is_blunder") == 0)

    print(f"\nDone!")
    print(f"Games processed: {game_count} | Skipped: {skipped}")
    print(f"Total moves: {len(all_moves)}")
    print(f"Blunders: {blunders} ({100*blunders/len(all_moves):.1f}%)")
    print(f"Non-blunders: {non_blunders}")

    with open(output_path, "w") as f:
        json.dump(all_moves, f)

    print(f"\nSaved to {output_path}")