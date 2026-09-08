import asyncio,sys,json
import websockets
url=sys.argv[1]
async def main():
    async with websockets.connect(url,origin="http://127.0.0.1:3004",open_timeout=20) as ws:
        print("WEBSOCKET HANDSHAKE : PASS")
        await ws.send(json.dumps({"type":"PING"}))
        print("WEBSOCKET RESPONSE  :",await asyncio.wait_for(ws.recv(),timeout=10))
asyncio.run(main())
