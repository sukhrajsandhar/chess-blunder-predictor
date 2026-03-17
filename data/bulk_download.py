import requests
import os
import time
import chess.pgn
import io

def download_player_games_pgn(username, max_games=200):
    """
    Correct Lichess API call per docs:
    GET https://lichess.org/api/games/user/{username}
    Accept: application/x-chess-pgn
    """
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "clocks": "true",
        "evals": "true",
        "max": max_games,
        "perfType": "rapid,classical",
        "rated": "true",
    }
    headers = {
        "Accept": "application/x-chess-pgn",
    }

    print(f"Downloading {username}...")
    r = requests.get(url, params=params, headers=headers)

    if r.status_code != 200:
        print(f"  Failed: {r.status_code}")
        return ""

    # Filter to only keep games that actually have evals
    pgn_text = r.text
    games_with_evals = []
    pgn_io = io.StringIO(pgn_text)

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        # Check if this game has eval annotations
        game_str = str(game)
        if "[%eval" in game_str:
            games_with_evals.append(game_str)

    print(f"  Total: {pgn_text.count('[Event')} games, with evals: {len(games_with_evals)}")
    return "\n\n".join(games_with_evals)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    # Use DDT3000 — confirmed working from Lichess forum post
    # Plus other players known to have analyzed games
    players = [
        "DDT3000",
        "lance5500",
        "lovlas",
        "thibault",
        "zhigalko_sergei",
        "Penguingim1",
        "chess-network",
        "jbacon",
        "somethingpretentious",
        "revoof",
        "Beneficent",
        "cyanfish",
        "isaaccsas",
        "MagnusEffect",
        "ATorre1920",
    ]

    all_pgn = ""
    total = 0

    for player in players:
        pgn = download_player_games_pgn(player, max_games=200)
        count = pgn.count("[Event")
        all_pgn += "\n\n" + pgn
        total += count
        print(f"  Running total: {total} games with evals")
        time.sleep(2)

    with open("data/raw/games.pgn", "w", encoding="utf-8") as f:
        f.write(all_pgn)

    print(f"\nDone. {total} games with evals saved to data/raw/games.pgn")