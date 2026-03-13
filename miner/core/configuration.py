import os
from functools import lru_cache
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

from fiber.chain import chain_utils, interface
from fiber.chain.metagraph import Metagraph
from fiber.miner.security.nonce_management import NonceManager
from miner.core.models.config import Config

T = TypeVar("T", bound=BaseModel)

load_dotenv()

@lru_cache
def factory_config() -> Config:
    # Initialize security components
    nonce_manager = NonceManager()

    # Load fiber network configuration
    wallet_name = os.getenv("WALLET_NAME", "default")
    hotkey_name = os.getenv("HOTKEY_NAME", "default")
    netuid = os.getenv("NETUID")
    subtensor_network = os.getenv("SUBTENSOR_NETWORK")
    subtensor_address = os.getenv("SUBTENSOR_ADDRESS")
    load_old_nodes = bool(os.getenv("LOAD_OLD_NODES", True))
    min_stake_threshold = float(os.getenv("MIN_STAKE_THRESHOLD", 1000))
    refresh_nodes = os.getenv("REFRESH_NODES", "true").lower() == "true"
    device = os.getenv("DEVICE", "cpu")

    assert netuid is not None, "Must set NETUID env var please!"

    if refresh_nodes:
        substrate = interface.get_substrate(subtensor_network, subtensor_address)
        metagraph = Metagraph(
            substrate=substrate,
            netuid=netuid,
            load_old_nodes=load_old_nodes,
        )
    else:
        metagraph = Metagraph(substrate=None, netuid=netuid, load_old_nodes=load_old_nodes)

    keypair = chain_utils.load_hotkey_keypair(wallet_name, hotkey_name)

    return Config(
        # Device config
        device=device,
        
        # Network and security config
        nonce_manager=nonce_manager,
        keypair=keypair,
        metagraph=metagraph,
        min_stake_threshold=min_stake_threshold,
        httpx_client=httpx.AsyncClient()
    ) 