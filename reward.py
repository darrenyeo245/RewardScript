import threading
import numpy as np
from pythonosc import osc_server, udp_client
from pythonosc.dispatcher import Dispatcher

IN_IP = "0.0.0.0"
IN_PORT = 9006
BROADCAST_IP = "127.0.0.1"
BROADCAST_PORT = 8000

OUT_ADDRESS = "/adm/obj/101/xyz"
REWARD_ADDRESS = "/reward"
IN_ADDRESS = "/adm/obj/1/xyz"
RESET_ADDRESS = ("/episode/end", "/episode/reset_manual")
STOP_ADDR = "/training/stop"

MAX_DIST = float(np.sqrt(12.0))

lock = threading.Lock()
client = udp_client.SimpleUDPClient(BROADCAST_IP, BROADCAST_PORT, allow_broadcast=True)

output = np.zeros(3, dtype=np.float32)
inputs = {IN_ADDRESS: np.zeros(3, dtype=np.float32)}
reward_value = 0.0

pending_output = None
stopped = False
new_input_since_last_reward = False

RNG = np.random.default_rng()
STEP_SCALE = 0.03
DRIFT_SCALE = 0.02
Z_BIAS = 0.0


def clamp_xyz(values):
    arr = np.array(values[:3], dtype=np.float32)
    return np.clip(arr, -1.0, 1.0)


def distance_to_reward(dist):
    x = min(max(dist / MAX_DIST, 0.0), 1.0)
    return 1.0 - 2.0 * x


def send_output(advance=False, reset=False):
    global output

    if stopped:
        return

    if reset:
        output = np.zeros(3, dtype=np.float32)
    elif advance:
        delta = RNG.normal(0.0, STEP_SCALE, size=3)
        candidate = clamp_xyz(output + delta)
        alpha = 0.05
        output = clamp_xyz((1.0 - alpha) * output + alpha * candidate)

    payload = output.tolist()
    client.send_message(OUT_ADDRESS, payload)
    log_tx(OUT_ADDRESS, payload)

def send_reward():
    if stopped:
        return
    payload = float(reward_value)
    client.send_message(REWARD_ADDRESS, payload)
    log_tx(REWARD_ADDRESS, payload)


def compute_reward():
    global reward_value

    inp = inputs[IN_ADDRESS]  # nur ein Objekt verwenden
    dist = np.linalg.norm(inp - output)
    reward_value = float(distance_to_reward(dist))


def on_input(address, *args):
    log_rx(address, args)

    if len(args) < 3:
        return
    xyz = clamp_xyz(args)

    with lock:
        inputs[address] = xyz

        compute_reward()
        send_reward()
        send_output(advance=True)

def step_handler(address, *args):
    del args
    log_rx(address, [])
    global output, pending_output

    with lock:
        if stopped:
            return

        if pending_output is not None:
            output = pending_output
            pending_output = None
        else:
            noise = RNG.normal(0.0, STEP_SCALE, size=3)
            drift = -output * DRIFT_SCALE
            drift[2] = abs(drift[2]) + Z_BIAS
            output = clamp_xyz(output + noise + drift)

        send_output()

def reset_handler(address, *args):
    del args
    log_rx(address, [])

    with lock:
        if stopped:
            return
        send_output(reset=True)

def stop_handler(address, *args):
    del args
    global stopped
    with lock:
        stopped = True
    log_rx(address, [])
    print("[INFO] training_stop empfangen -> Senden gestoppt.")

def log_rx(address, args):
    print(f"[RX] {address} {list(args)}")

def log_tx(address, payload):
    print(f"[TX] {address} {payload}")


dispatcher = Dispatcher()
for addr in RESET_ADDRESS:
    dispatcher.map(addr, reset_handler)
dispatcher.map(IN_ADDRESS, on_input)
dispatcher.map(STOP_ADDR, stop_handler)

server = osc_server.ThreadingOSCUDPServer((IN_IP, IN_PORT), dispatcher)
print(f"Reward engine listening on {IN_IP}:{IN_PORT}")

with lock:
    compute_reward()
    send_reward()
    send_output()

server.serve_forever()