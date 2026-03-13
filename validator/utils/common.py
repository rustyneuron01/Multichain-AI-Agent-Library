import os
from typing import List
from fiber.logging_utils import get_logger
from fiber.chain.models import Node
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.chain.interface import get_substrate
from dotenv import load_dotenv
from fiber.validator.client import construct_server_address, make_non_streamed_get
from fiber.validator.handshake import perform_handshake
import httpx
from cryptography.fernet import Fernet
import asyncio
from pathlib import Path

# Get the absolute path to .env
validator_dir = Path(__file__).parents[1]
env_path = validator_dir / ".env"

# Load only .env
load_dotenv(env_path)

logger = get_logger(__name__)

def truncate_log_data(data, max_length=50):
    """Helper function to truncate log data"""
    if isinstance(data, dict):
        return {k: truncate_log_data(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) > 10:
            return [truncate_log_data(v, max_length) for v in data[:10]] + [f"... and {len(data)-10} more items"]
        return [truncate_log_data(v, max_length) for v in data]
    elif isinstance(data, str) and len(data) > max_length:
        return data[:max_length] + '...'
    return data

def get_active_miners() -> List[Node]:
    """
    Get a list of active miners in the subnet.
    
    Returns:
        List[Node]: A list of Node objects containing miner information.
    """
    subtensor_network = os.getenv("SUBTENSOR_NETWORK", "local")
    subtensor_address = os.getenv("SUBTENSOR_ADDRESS", "ws://127.0.0.1:9946")
    netuid = int(os.getenv("NETUID", "1"))
    
    logger.debug(f"Using subtensor network: {subtensor_network}")
    logger.debug(f"Using subtensor address: {subtensor_address}")
    logger.debug(f"Using netuid: {netuid}")
    
    substrate = get_substrate(subtensor_network=subtensor_network, subtensor_address=subtensor_address)
    logger.debug(f"Connected to substrate at: {substrate.url}")
    
    active_miners = get_nodes_for_netuid(substrate, netuid)
    
    logger.info(f"Retrieved {len(active_miners)} active miners")
    for miner in active_miners:
        logger.debug(f"Found miner - Hotkey: {miner.hotkey}, IP: {miner.ip}, Port: {miner.port}")
    return active_miners

async def check_node_availability(client: httpx.AsyncClient, server_address: str, validator_ss58_address: str, fernet: Fernet, symmetric_key_uuid: str) -> bool:
    try:
        response = await make_non_streamed_get(
            httpx_client=client,
            server_address=server_address,
            endpoint="/availability",
            validator_ss58_address=validator_ss58_address,
            symmetric_key_uuid=symmetric_key_uuid,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('available', False)
        else:
            logger.warning(f"Failed to check availability for {server_address}: Status code {response.status_code}")
            return False
    except httpx.ConnectTimeout:
        logger.warning(f"Connection timeout checking availability for {server_address}")
        return False
    except Exception as e:
        logger.error(f"Error checking availability for {server_address}: {str(e)}")
        return False

async def perform_multiple_handshakes(nodes, client, keypair):
    async def single_handshake(node):
        # Use VALIDATOR_HOST from env to determine if we should use localhost
        validator_host = os.getenv('VALIDATOR_HOST', '0.0.0.0')
        replace_localhost = validator_host in ['0.0.0.0', 'localhost', '127.0.0.1']
        
        server_address = construct_server_address(node, replace_with_localhost=replace_localhost)
        #logger.debug(f"Attempting handshake with node at {server_address} (replace_localhost={replace_localhost})")
        
        try:
            # Attempt handshake with shorter timeout
            symmetric_key_str, symmetric_key_uuid = await perform_handshake(
                client,
                server_address,
                keypair,
                node.hotkey
            )
            fernet = Fernet(symmetric_key_str)
            
            # Now check if node is available using the encryption keys
            if not await check_node_availability(
                client, 
                server_address, 
                keypair.ss58_address,
                fernet,
                symmetric_key_uuid
            ):
                logger.warning(f"Node {server_address} is not available")
                return node.node_id, None
            
            logger.info(f"Handshake successful with node {node.node_id} at {server_address}")
            return node.node_id, (fernet, symmetric_key_uuid, server_address, node.hotkey)
        except httpx.ConnectTimeout:
            logger.warning(f"Connection timeout during handshake with {server_address}")
            return node.node_id, None
        except Exception as e:
            logger.error(f"Handshake failed with {server_address}: {str(e)}", exc_info=True)
            return node.node_id, None

    handshake_tasks = [single_handshake(node) for node in nodes]
    results = await asyncio.gather(*handshake_tasks)
    
    # Filter out failed handshakes and create the node_ferners dict
    node_ferners = {node_id: data for node_id, data in results if data is not None}
    return node_ferners

async def refresh_nodes(substrate):
    """
    Refresh the list of active nodes in the subnet.
    
    Args:
        substrate: The substrate instance to use for querying the chain.
    """
    netuid = int(os.getenv("NETUID", "1"))
    try:
        nodes = get_nodes_for_netuid(substrate, netuid)
        logger.info(f"Found {len(nodes)} registered nodes")
        return nodes
    except Exception as e:
        logger.error(f"Failed to refresh nodes: {str(e)}")
        return []
