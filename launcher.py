"""
launcher.py - CLI launcher for app.py

Retries launching app.py if it throws KeyboardInterrupts.
"""

import sys
import time
import traceback

MAX_RETRIES = 3
RETRY_DELAY = 1.0


def run_app():  # wrapper function stops leaky retries
    import app

    app.main()


def main():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            run_app()
            return

        except KeyboardInterrupt:
            print(
                f"\nKeyboardInterrupt during startup (attempt {attempt}/{MAX_RETRIES})"
            )

            if attempt == MAX_RETRIES:
                print("Max retries reached. Exiting.")
                sys.exit(1)

            print(f"Retrying in {RETRY_DELAY} seconds...\n")
            time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"\nUnhandled error (attempt {attempt}/{MAX_RETRIES}): {e}")
            traceback.print_exc()

            if attempt == MAX_RETRIES:
                print("Max retries reached. Exiting.")
                sys.exit(1)

            print(f"Retrying in {RETRY_DELAY} seconds...\n")
            time.sleep(RETRY_DELAY)


if __name__ == "__main__":
    main()
