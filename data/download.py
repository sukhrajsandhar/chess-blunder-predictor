# A rapid game with clear blunders for testing
import requests

def download_game(game_id):
    url = f"https://lichess.org/game/export/{game_id}"
    params = {"clocks": True, "evals": True}
    headers = {"Accept": "application/x-chess-pgn"}
    r = requests.get(url, params=params, headers=headers)
    if r.status_code == 200:
        with open(f"data/{game_id}.pgn", "w") as f:
            f.write(r.text)
        print(f"Downloaded {game_id}")
        print(r.text[:500])
    else:
        print(f"Failed: {r.status_code}")

# Magnus Carlsen rapid game with clear blunders
download_game("HbKF7UPv")