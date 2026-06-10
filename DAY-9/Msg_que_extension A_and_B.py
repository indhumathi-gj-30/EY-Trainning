import json
import asyncio
from typing import Any, Dict
import aio_pika
from azure.servicebus import ServiceBusClient, ServiceBusMessage

AMQP_CLOUD_ENDPOINT = "amqps://<username>:<password>@<host>/<vhost>"
AZURE_BUS_TOKEN_STR = "<AZURE_SERVICE_BUS_CONNECTION_STRING>"
TRANSACTION_ROUTE_KEY = "payments"
session_connection = None
session_channel = None
transaction_inbox_queue = None

async def establish_amqp_session():
    """Initializes highly robust connection clusters to cloud instance."""
    global session_connection, session_channel, transaction_inbox_queue

    session_connection = await aio_pika.connect_robust(AMQP_CLOUD_ENDPOINT)
    session_channel = await session_connection.channel()

    # Poison Pillar Dead Letter Infrastructure
    fault_exchange = await session_channel.declare_exchange(
        "dlx",
        aio_pika.ExchangeType.DIRECT
    )

    fault_isolation_queue = await session_channel.declare_queue(
        "payments.dlq",
        durable=True
    )

    await fault_isolation_queue.bind(
        fault_exchange,
        routing_key="payments.failed"
    )

    # Core Transaction Data Ingestion Queue
    transaction_inbox_queue = await session_channel.declare_queue(
        TRANSACTION_ROUTE_KEY,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "dlx",
            "x-dead-letter-routing-key": "payments.failed"
        }
    )

    print("Successfully synchronized with CloudAMQP Broker cluster topology.")


async def dispatch_amqp_payload(event_data: Dict[str, Any]):
    """Packages and streams structured events downstream to queue lines."""
    serialized_bytes = json.dumps(event_data).encode()
    
    await session_channel.default_exchange.publish(
        aio_pika.Message(body=serialized_bytes),
        routing_key=TRANSACTION_ROUTE_KEY
    )

    print("Telemetry transaction dispatched successfully to AMQP Cluster.")


async def process_amqp_stream():
    """Continuously intercepts incoming stream nodes via a dynamic iterator loop."""
    async with transaction_inbox_queue.iterator() as runtime_stream:
        async for active_packet in runtime_stream:
            async with active_packet.process():
                unpacked_payload = json.loads(active_packet.body.decode())
                print("AMQP Interceptor Context Processing Unit:", unpacked_payload)

def dispatch_azure_bus_payload(event_data: Dict[str, Any]):
    """Sync transaction injection inside Azure enterprise pipeline context."""
    with ServiceBusClient.from_connection_string(AZURE_BUS_TOKEN_STR) as runtime_client:
        message_dispatcher = runtime_client.get_queue_sender(
            queue_name=TRANSACTION_ROUTE_KEY
        )
        with message_dispatcher:
            stringified_json = json.dumps(event_data)
            message_dispatcher.send_messages(
                ServiceBusMessage(stringified_json)
            )

    print("Telemetry transaction dispatched successfully to Azure Cloud Service Bus.")


def process_azure_bus_stream():
    """Polls downstream messages from specific target data partitions."""
    with ServiceBusClient.from_connection_string(AZURE_BUS_TOKEN_STR) as runtime_client:
        message_consumer = runtime_client.get_queue_receiver(
            queue_name=TRANSACTION_ROUTE_KEY
        )
        with message_consumer:
            fetched_packets = message_consumer.receive_messages(
                max_message_count=1,
                max_wait_time=5
            )

            for targeted_packet in fetched_packets:
                print("Azure Bus Interceptor Context Processing Unit:", str(targeted_packet))
                message_consumer.complete_message(targeted_packet)


def quarantine_faulty_azure_packet():
    """Moves invalid transaction contexts to specialized isolation vectors."""
    with ServiceBusClient.from_connection_string(AZURE_BUS_TOKEN_STR) as runtime_client:
        message_consumer = runtime_client.get_queue_receiver(
            queue_name=TRANSACTION_ROUTE_KEY
        )
        with message_consumer:
            fetched_packets = message_consumer.receive_messages(
                max_message_count=1,
                max_wait_time=5
            )

            for targeted_packet in fetched_packets:
                message_consumer.dead_letter_message(
                    targeted_packet,
                    reason="Payment Processing Failed"
                )
                print("Active transaction quarantined and moved to Azure DLQ storage vector.")

simulated_ledger_packet = {
    "amount": 500,
    "currency": "GBP",
    "account_id": "ACC-001",
    "reference": "Invoice-1001"
}

async def orchestration_pipeline_trigger():
    # Execute AMQP Cluster Node Sequence
    await establish_amqp_session()
    await dispatch_amqp_payload(simulated_ledger_packet)
    
    # Executing listener script with a safe tracking window
    try:
        await asyncio.wait_for(process_amqp_stream(), timeout=2.0)
    except asyncio.TimeoutError:
        print("AMQP local stream interceptor suspended safely (Timeout monitoring block).")

    # Execute Azure Enterprise Service Bus Sequence
    dispatch_azure_bus_payload(simulated_ledger_packet)
    process_azure_bus_stream()
    quarantine_faulty_azure_packet()

    print("Architecture Extension Cluster A: AMQP Messaging Services Activated.")
    print("Architecture Extension Cluster B: Azure Service Bus Engine Activated.")

if __name__ == "__main__":
    asyncio.run(orchestration_pipeline_trigger())