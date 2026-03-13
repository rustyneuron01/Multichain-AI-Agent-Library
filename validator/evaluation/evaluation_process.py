
import os
import asyncio
from loguru import logger
from validator.evaluation.evaluation_loop import run_evaluation_loop

def start_evaluation():
    logger.info("[EVAL_PROCESS] Starting evaluation subprocess...")

    try:
        db_path = os.getenv("DB_PATH")
        api_key = os.getenv("OPENAI_API_KEY")
        hotkey = os.getenv("VALIDATOR_HOTKEY")

        if not all([db_path, api_key, hotkey]):
            raise ValueError(f"[EVAL_PROCESS] Missing one of DB_PATH={db_path}, API_KEY={api_key}, HOTKEY={hotkey}")

        logger.info(f"[EVAL_PROCESS] ENV OK - db_path={db_path}, hotkey={hotkey}")

        asyncio.run(
            run_evaluation_loop(
                db_path=db_path,
                openai_api_key=api_key,
                validator_hotkey=hotkey,
                batch_size=10,
                sleep_interval=120
            )
        )
    
    except Exception as e:
        logger.exception(f"[EVAL_PROCESS] Subprocess crashed: {e}")
