"""Update the Spectre Studio index from the terminal."""

from study_chat import AppError, rebuild


if __name__ == "__main__":
    try:
        result = rebuild(progress_callback=print)
    except AppError as error:
        raise SystemExit(str(error)) from error
    print(result["message"])
