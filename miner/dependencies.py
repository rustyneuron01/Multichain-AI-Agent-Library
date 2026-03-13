from fastapi import Depends, Header, HTTPException, Request

from fiber import constants as cst
from fiber import utils
from fiber.chain import signatures
from fiber.logging_utils import get_logger
from miner.core import configuration
from miner.core.models.config import Config

logger = get_logger(__name__)


def get_config() -> Config:
    return configuration.factory_config()


async def verify_request(
    request: Request,
    validator_hotkey: str = Header(..., alias=cst.VALIDATOR_HOTKEY),
    signature: str = Header(..., alias=cst.SIGNATURE),
    miner_hotkey: str = Header(..., alias=cst.MINER_HOTKEY),
    nonce: str = Header(..., alias=cst.NONCE),
    config: Config = Depends(get_config),
):
    if not config.nonce_manager.nonce_is_valid(nonce):
        logger.debug("Nonce is not valid!")
        raise HTTPException(
            status_code=401,
            detail="Oi, that nonce is not valid!",
        )

    body = await request.body()
    payload_hash = signatures.get_hash(body)
    message = utils.construct_header_signing_message(nonce=nonce, miner_hotkey=miner_hotkey, payload_hash=payload_hash)
    if not signatures.verify_signature(
        message=message,
        signer_ss58_address=validator_hotkey,
        signature=signature,
    ):
        raise HTTPException(
            status_code=401,
            detail="Oi, invalid signature, you're not who you said you were!",
        )


async def blacklist_low_stake(
    validator_hotkey: str = Header(..., alias=cst.VALIDATOR_HOTKEY), config: Config = Depends(get_config)
):
    metagraph = config.metagraph
    metagraph.sync_nodes()
    node = metagraph.nodes.get(validator_hotkey)
    logger.info(f"Node {validator_hotkey} has TAO stake {node.tao_stake}")
    logger.info(f"Node full object: {node}")
    if not node:
        raise HTTPException(status_code=403, detail="Hotkey not found in metagraph")

    if node.tao_stake < config.min_stake_threshold:
        logger.debug(f"Node {validator_hotkey} has insufficient stake of {node.tao_stake} - minimum is {config.min_stake_threshold}")
        raise HTTPException(status_code=403, detail=f"Insufficient stake of {node.tao_stake} - minimum is {config.min_stake_threshold}  ") 
   