import time
from pathlib import Path

import pandas as pd
import requests


TOKEN_URL = "https://cxm-api.fifa.com/fifaplusweb/api/external/gameDay/token"
STATS_URL = "https://gameday-prod.fifa.mangodev.co.uk/1-0/stories"
PLAYER_URL = "https://api.fifa.com/api/v3/players/{}"

COMPETITION_ID = "285023"
OUTPUT_FILE = Path(__file__).with_name("fifa_attacking_distribution.xlsx")

REQUEST_TIMEOUT = (10, 30)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.fifa.com/",
}


def request_json(session, url, headers=None, params=None, retries=3):
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                print("Status:", response.status_code)
                print("URL:", response.url)
                print(response.text[:500])
                response.raise_for_status()

            return response.json()

        except Exception as error:
            print(f"Request error {attempt}/{retries}: {error}")

            if attempt == retries:
                raise

            time.sleep(2 * attempt)


def tags_to_dict(tags):
    result = {}

    for tag in tags:
        if isinstance(tag, dict) and "name" in tag:
            result[tag["name"]] = tag.get("value")

    return result


def get_token(session):
    print("Getting FIFA token...")
    data = request_json(session, TOKEN_URL, headers=HEADERS)
    return data["token"]


def get_story(session, token, external_id):
    headers = HEADERS.copy()
    headers["Authorization"] = f"Bearer {token}"

    query = (
        "(and "
        "resourceStatus==`urn:gd:resourceStatus:active` "
        f"_externalId==`{external_id}`)"
    )

    params = {
        "query": query,
        "skip": 0,
        "limit": 1,
    }

    data = request_json(
        session,
        STATS_URL,
        headers=headers,
        params=params,
    )

    items = data.get("items", [])

    if not items:
        raise RuntimeError(f"No story found: {external_id}")

    return items[0]


def get_player_id(actor):
    key = actor.get("key", {})

    if isinstance(key, dict):
        player_id = key.get("_externalSportsPersonId")
    else:
        player_id = None

    if not player_id:
        player_id = actor.get("id")

    return str(player_id) if player_id else ""


def get_player_name(actor):
    name = actor.get("name", "")

    if isinstance(name, dict):
        return name.get("eng", "")

    return str(name)


def get_basic_info(actor):
    actor_tags = tags_to_dict(actor.get("tags", []))

    return {
        "Player ID": get_player_id(actor),
        "Player": get_player_name(actor),
        "Nation": actor_tags.get(
            "urn:gd:tag:story:team:abbreviation",
            "",
        ),
        "Position": actor_tags.get(
            "urn:gd:tag:story:staff:position",
            "",
        ),
    }


def collect_stat_section(session, token, classification, section_name, story_prefix):
    first_external_id = f"{story_prefix}:page:1"
    first_story = get_story(session, token, first_external_id)

    story_tags = tags_to_dict(first_story.get("tags", []))

    column_order_tag = f"urn:gd:tag:story:{classification}:column_order"

    stat_tags = story_tags[column_order_tag]
    column_titles = story_tags["urn:gd:tag:story:stat:column_titles:eng"]

    column_mapping = dict(zip(stat_tags, column_titles))

    output_columns = [
        f"{section_name} - {title}"
        for title in column_titles
    ]

    page_count = int(story_tags["urn:gd:tag:story:page_count"])

    players = {}

    for page_number in range(1, page_count + 1):
        print(f"Downloading {section_name}: page {page_number}/{page_count}")

        if page_number == 1:
            story = first_story
        else:
            external_id = f"{story_prefix}:page:{page_number}"
            story = get_story(session, token, external_id)

        for actor in story.get("actors", []):
            player_id = get_player_id(actor)

            if not player_id:
                continue

            actor_tags = tags_to_dict(actor.get("tags", []))
            player = get_basic_info(actor)

            for stat_tag, title in column_mapping.items():
                column_name = f"{section_name} - {title}"
                player[column_name] = actor_tags.get(stat_tag, "")

            players[player_id] = player

        time.sleep(0.1)

    return players, output_columns


def collect_minutes_played(session, token):
    classification = "gcp_top_scorer"

    story_prefix = (
        "urn:gd:story:classification:gcp_top_scorer:"
        f"competitionId:{COMPETITION_ID}:goals:rank_asc"
    )

    first_story = get_story(session, token, f"{story_prefix}:page:1")
    story_tags = tags_to_dict(first_story.get("tags", []))
    page_count = int(story_tags["urn:gd:tag:story:page_count"])

    players = {}

    for page_number in range(1, page_count + 1):
        print(f"Downloading Minutes Played: page {page_number}/{page_count}")

        if page_number == 1:
            story = first_story
        else:
            story = get_story(
                session,
                token,
                f"{story_prefix}:page:{page_number}",
            )

        for actor in story.get("actors", []):
            player_id = get_player_id(actor)

            if not player_id:
                continue

            actor_tags = tags_to_dict(actor.get("tags", []))
            player = get_basic_info(actor)

            player["Minutes Played"] = actor_tags.get(
                "urn:gd:tag:football:stats:total_competition_minutes_played",
                "",
            )

            players[player_id] = player

        time.sleep(0.1)

    return players


def get_birth_date(session, player_id):
    url = PLAYER_URL.format(player_id)

    for attempt in range(1, 4):
        try:
            response = session.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                birth_date = data.get("BirthDate", "")

                if birth_date:
                    return birth_date.split("T")[0]

                return ""

            print(
                f"Birthday request failed for {player_id}: "
                f"{response.status_code}"
            )

        except Exception as error:
            print(f"Birthday error {attempt}/3 for {player_id}: {error}")

        time.sleep(1 * attempt)

    return ""


def choose_first_value(*values):
    for value in values:
        if value not in [None, "", "N/A"]:
            return value

    return ""


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    token = get_token(session)

    attacking_prefix = (
        "urn:gd:story:classification:gcp_attack:"
        f"competitionId:{COMPETITION_ID}:assists:rank_asc"
    )

    distribution_prefix = (
        "urn:gd:story:classification:gcp_distribution:"
        f"competitionId:{COMPETITION_ID}:passes:rank_asc"
    )

    attacking, attacking_columns = collect_stat_section(
        session=session,
        token=token,
        classification="gcp_attack",
        section_name="Attacking",
        story_prefix=attacking_prefix,
    )

    distribution, distribution_columns = collect_stat_section(
        session=session,
        token=token,
        classification="gcp_distribution",
        section_name="Distribution",
        story_prefix=distribution_prefix,
    )

    minutes = collect_minutes_played(session, token)

    player_ids = sorted(
        set(attacking)
        | set(distribution)
        | set(minutes)
    )

    rows = []

    print(f"Getting birthdays for {len(player_ids)} players...")

    for number, player_id in enumerate(player_ids, start=1):
        attack_player = attacking.get(player_id, {})
        distribution_player = distribution.get(player_id, {})
        minutes_player = minutes.get(player_id, {})

        row = {
            "Player ID": player_id,
            "Player": choose_first_value(
                attack_player.get("Player"),
                distribution_player.get("Player"),
                minutes_player.get("Player"),
            ),
            "Nation": choose_first_value(
                attack_player.get("Nation"),
                distribution_player.get("Nation"),
                minutes_player.get("Nation"),
            ),
            "Position": choose_first_value(
                attack_player.get("Position"),
                distribution_player.get("Position"),
                minutes_player.get("Position"),
            ),
            "Birth Date": get_birth_date(session, player_id),
            "Minutes Played": minutes_player.get("Minutes Played", ""),
        }

        for column in attacking_columns:
            row[column] = attack_player.get(column, "")

        for column in distribution_columns:
            row[column] = distribution_player.get(column, "")

        rows.append(row)

        if number % 25 == 0 or number == len(player_ids):
            print(f"Birthdays done: {number}/{len(player_ids)}")

        time.sleep(0.03)

    df = pd.DataFrame(rows)

    column_order = (
        [
            "Player ID",
            "Player",
            "Nation",
            "Position",
            "Birth Date",
            "Minutes Played",
        ]
        + attacking_columns
        + distribution_columns
    )

    df = df[column_order]
    df.sort_values(["Nation", "Player"], inplace=True)

    try:
        df.to_excel(
            OUTPUT_FILE,
            sheet_name="Player Statistics",
            index=False,
            engine="openpyxl",
        )

        print(f"Saved: {OUTPUT_FILE}")
        print(f"Total players: {len(df)}")

    except PermissionError:
        print("Cannot save file. Close the Excel file first, then run again.")


if __name__ == "__main__":
    main()