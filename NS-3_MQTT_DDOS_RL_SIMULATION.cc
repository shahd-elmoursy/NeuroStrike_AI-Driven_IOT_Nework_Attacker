/* =============================================================================
 * mqtt_ddos_rl_simulation.cc
 *
 * NS-3 recreation of the real hardware testbed
 * =============================================
 *
 * REAL HARDWARE (what this simulates):
 * ─────────────────────────────────────
 *  One WiFi network — all devices on the same subnet
 *
 *  ESP32 #1  → Sensor0  + Sensor1   ──┐
 *  ESP32 #2  → Sensor2  + Sensor3   ──┤──► Broker0 (192.168.0.133)
 *  ESP32 #3  → Sensor4  + Sensor5   ──┘
 *
 *  ESP32 #4  → Sensor6  + Sensor7   ──┐
 *  ESP32 #5  → Sensor8  + Sensor9   ──┤──► Broker1 (192.168.0.134)
 *  ESP32 #6  → Sensor10 + Sensor11  ──┘
 *
 *  Raspberry Pi (192.168.0.109)
 *    → RL agent picks ONE attack mode at a time
 *    → Spoofs ESP32 sensor IPs when attacking
 *    → Attacks both brokers (not necessarily simultaneously)
 *    → Packet sizes follow CTGAN distribution (random, not fixed)
 *    → Goal: blend in with sensor traffic so broker can't detect it
 *
 * NS-3 TOPOLOGY:
 * ──────────────
 *  Single 802.11b WiFi ad-hoc network  192.168.0.0/24
 *  All nodes on the same channel — mirrors real hardware
 *
 *  Node assignments:
 *    192.168.0.133  Broker0
 *    192.168.0.134  Broker1
 *    192.168.0.153  Sensor0  + spoofed by attacker  (ESP32 #1, slot A)
 *    192.168.0.154  Sensor1                          (ESP32 #1, slot B)
 *    192.168.0.160  Sensor2  + spoofed by attacker  (ESP32 #2, slot A)
 *    192.168.0.161  Sensor3                          (ESP32 #2, slot B)
 *    192.168.0.176  Sensor4  + spoofed by attacker  (ESP32 #3, slot A)
 *    192.168.0.177  Sensor5                          (ESP32 #3, slot B)
 *    192.168.0.178  Sensor6  + spoofed by attacker  (ESP32 #4, slot A)
 *    192.168.0.179  Sensor7                          (ESP32 #4, slot B)
 *    192.168.0.180  Sensor8  + spoofed by attacker  (ESP32 #5, slot A)
 *    192.168.0.181  Sensor9                          (ESP32 #5, slot B)
 *    192.168.0.182  Sensor10 + spoofed by attacker  (ESP32 #6, slot A)
 *    192.168.0.183  Sensor11                         (ESP32 #6, slot B)
 *    192.168.0.109  Raspberry Pi (attacker)
 *
 * SPOOFING IN NS-3:
 * ─────────────────
 *  NS-3 does not support raw-socket IP spoofing at the application layer.
 *  We model spoofing by having the attacker node cycle through the sensor
 *  IP addresses as its SOURCE address for each attack connection.
 *  This is done by binding a new socket per attack packet to one of the
 *  spoofed IPs before connecting — matching exactly what NeuroStrike does
 *  with scapy on the real Pi.
 *
 * MQTT FRAMES:
 * ────────────
 *  Real MQTT 3.1.1 binary frames over TCP port 1883.
 *  Wireshark filter "mqtt" will show CONNECT / CONNACK / PUBLISH.
 *  Attack frames also use real MQTT structure matching the 5 CTGAN modes.
 *
 * PACKET SIZES:
 * ─────────────
 *  Sensor traffic: sizes from real PCAP (normal_traffic_18thfeb.pcapng)
 *  Attack traffic: sizes sampled randomly from CTGAN distribution per mode
 *    not fixed to the mean — uses gaussian(mean, std) from model statistics
 *
 * CTGAN ATTACK MODES (RL agent picks one at a time):
 * ────────────────────────────────────────────────────
 *  0  SYN_TCP_Flooding                 mean=62B  std=4B   delta=1.709s
 *  1  Basic_Connect_Flooding           mean=61B  std=4B   delta=0.476s
 *  2  Delayed_Connect_Flooding         mean=62B  std=4B   delta=0.890s
 *  3  Invalid_Subscription_Flooding    mean=63B  std=4B   delta=0.751s
 *  4  Connect_Flooding_with_WILL       mean=162B std=20B  delta=0.490s
 *
 * RL AGENT (Q-learning, matches neurostrike_pi_v2):
 * ──────────────────────────────────────────────────
 *  State:   packet loss ratio from FlowMonitor
 *  Reward:  1.0 - 2.0 * loss_ratio  (low loss = attack effective)
 *  Update:  Q[m] += LR * (reward + GAMMA * max(Q) - Q[m])
 *  Policy:  epsilon-greedy  eps=0.90 → 0.05, decay=0.98/step
 *  Window:  EVAL_INTERVAL seconds between evaluations
 *  Target:  alternates between Broker0 and Broker1
 *
 * OUTPUT → rl_output/
 * ────────────────────
 *  pcap/broker0-*.pcap      open in Wireshark → filter: mqtt
 *  pcap/broker1-*.pcap
 *  pcap/attacker-*.pcap
 *  flows.csv                per-flow statistics
 *  rl_log.csv               every RL decision (mode, reward, Q-table)
 *  timeseries.csv           per-second bandwidth by category
 *  summary.csv              totals + final Q-table
 *  mqtt-rl-anim.xml         NetAnim animation
 *  mqtt-rl-flowmonitor.xml  FlowMonitor XML
 *
 * BUILD & RUN:
 * ────────────
 *  cp mqtt_ddos_rl_simulation.cc $NS3_DIR/scratch/
 *  cd $NS3_DIR && ./ns3 build
 *  ./ns3 run scratch/mqtt_ddos_rl_simulation
 *
 * CONFIGURABLE PARAMETERS (command line):
 *  --simTime=90          total simulation seconds
 *  --attackStart=10      when attacker begins
 *  --evalInterval=8      RL evaluation window (seconds)
 *  --epsilon=0.9         initial exploration rate
 *  --seed=42             random seed
 *  --outDir=rl_output    output folder
 *  --broker0Ip=...       Broker0 IP address
 *  --broker1Ip=...       Broker1 IP address
 *  --piIp=...            Raspberry Pi (attacker) IP
 * =============================================================================
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/netanim-module.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/inet-socket-address.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <sys/stat.h>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("MqttDDoSRLSimulation");

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURABLE NETWORK PARAMETERS
// All IPs are command-line configurable — nothing hardcoded
// ═══════════════════════════════════════════════════════════════════════════

static std::string g_broker0Ip    = "192.168.0.133";
static std::string g_broker1Ip    = "192.168.0.134";
static std::string g_piIp         = "192.168.0.109";

// Sensor IPs — 2 per ESP32, 6 ESP32s = 12 sensors
// Slot-A IPs (odd ESP32 sensors) are also the spoofed IPs used by attacker
static std::string g_sensorIps[12] = {
  "192.168.0.153", "192.168.0.154",   // ESP32 #1 → Broker0
  "192.168.0.160", "192.168.0.161",   // ESP32 #2 → Broker0
  "192.168.0.176", "192.168.0.177",   // ESP32 #3 → Broker0
  "192.168.0.178", "192.168.0.179",   // ESP32 #4 → Broker1
  "192.168.0.180", "192.168.0.181",   // ESP32 #5 → Broker1
  "192.168.0.182", "192.168.0.183",   // ESP32 #6 → Broker1
};

// Spoofed IPs = slot-A sensors (every even index: 0,2,4,6,8,10)
// matches IOT_IPS in neurostrike_pi_v2_final_withthroughput.py
static std::string g_spoofIps[6] = {
  "192.168.0.153", "192.168.0.160", "192.168.0.176",
  "192.168.0.178", "192.168.0.180", "192.168.0.182"
};

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════

static const uint16_t MQTT_PORT    = 1883;
static double      g_simTime       = 90.0;
static double      g_attackStart   = 10.0;
static double      g_evalInterval  = 8.0;
static double      g_epsilon       = 0.90;
static uint32_t    g_seed          = 42;
static std::string g_outDir        = "rl_output";

// RL hyperparameters — matching neurostrike_pi_v2
static const double EPS_DECAY  = 0.98;
static const double EPS_FLOOR  = 0.05;
static const double GAMMA      = 0.95;
static const double LR_RL      = 0.10;
static const int    N_MODES    = 5;

// ═══════════════════════════════════════════════════════════════════════════
// CTGAN ATTACK PROFILES
// mean/std from 200-sample statistics of each model
// Packet sizes are sampled randomly from Gaussian(mean, std) — not fixed
// This mimics the random packet length strategy to evade broker detection
// ═══════════════════════════════════════════════════════════════════════════

struct AttackProfile {
  const char *name;
  const char *shortName;
  double      pktMean;    // mean packet size (bytes) from CTGAN
  double      pktStd;     // std dev of packet size
  double      deltaTime;  // mean inter-packet time (seconds)
};

static const AttackProfile ATK[N_MODES] = {
  // name                              short    mean  std   delta
  { "SYN_TCP_Flooding",               "SYN",   62.0, 4.0,  1.709 },
  { "Basic_Connect_Flooding",         "BASIC",  61.0, 4.0,  0.476 },
  { "Delayed_Connect_Flooding",       "DELAY",  62.0, 4.0,  0.890 },
  { "Invalid_Subscription_Flooding",  "INVSUB", 63.0, 4.0,  0.751 },
  { "Connect_Flooding_with_WILL",     "WILL",  162.0,20.0,  0.490 },
};

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════════════

// Resolved NS-3 Ipv4Address objects (set after IP assignment)
static Ipv4Address g_b0, g_b1, g_pi;
static Ipv4Address g_sAddr[12];  // sensor addresses
static Ipv4Address g_spoof[6];   // spoofed addresses

// FlowMonitor
static Ptr<FlowMonitor>        g_mon;
static Ptr<Ipv4FlowClassifier> g_cls;

// ─── RL log ──────────────────────────────────────────────────────────────
struct RLEntry {
  double t; int step, prevMode, newMode;
  double loss, reward, eps, q[N_MODES];
  int    targetBroker; // 0 or 1
};
static std::vector<RLEntry> g_rlLog;

// ─── Timeseries (per-second snapshot) ────────────────────────────────────
struct TsEntry { double t; uint64_t sBytes, aBytes; };
static std::vector<TsEntry> g_ts;
static uint64_t g_prevSB = 0, g_prevAB = 0;

// ═══════════════════════════════════════════════════════════════════════════
// MQTT BINARY FRAME BUILDER
// Produces real MQTT 3.1.1 frames — Wireshark decodes these correctly
// ═══════════════════════════════════════════════════════════════════════════

namespace Mqtt {

static void vlen(std::vector<uint8_t> &o, uint32_t n) {
  do { uint8_t b=n&0x7F; n>>=7; o.push_back(b|(n?0x80:0)); } while(n);
}
static void mstr(std::vector<uint8_t> &o, const std::string &s) {
  o.push_back((s.size()>>8)&0xFF); o.push_back(s.size()&0xFF);
  for (char c:s) o.push_back((uint8_t)c);
}
static void app(std::vector<uint8_t> &d, const std::vector<uint8_t> &s) {
  d.insert(d.end(), s.begin(), s.end());
}

// CONNECT (0x10)
std::vector<uint8_t> Connect(const std::string &cid,
                              bool will=false,
                              const std::string &wt="",
                              const std::string &wm="") {
  std::vector<uint8_t> vh, pay, rem, f;
  mstr(vh,"MQTT"); vh.push_back(0x04);
  vh.push_back(will?0x06:0x02);
  vh.push_back(0x00); vh.push_back(0x3C);
  mstr(pay,cid);
  if(will){ mstr(pay,wt); mstr(pay,wm); }
  app(rem,vh); app(rem,pay);
  f.push_back(0x10); vlen(f,rem.size()); app(f,rem);
  return f;
}

// CONNACK (0x20)
std::vector<uint8_t> ConnAck() { return {0x20,0x02,0x00,0x00}; }

// PUBLISH (0x30) QoS 0
std::vector<uint8_t> Publish(const std::string &topic,
                              const std::string &payload) {
  std::vector<uint8_t> rem, f;
  mstr(rem,topic);
  for(char c:payload) rem.push_back((uint8_t)c);
  f.push_back(0x30); vlen(f,rem.size()); app(f,rem);
  return f;
}

// SUBSCRIBE (0x82)
std::vector<uint8_t> Subscribe(const std::string &topic, uint16_t pid=1) {
  std::vector<uint8_t> rem, f;
  rem.push_back((pid>>8)&0xFF); rem.push_back(pid&0xFF);
  mstr(rem,topic); rem.push_back(0x00);
  f.push_back(0x82); vlen(f,rem.size()); app(f,rem);
  return f;
}

Ptr<Packet> ToPkt(const std::vector<uint8_t> &v) {
  return Create<Packet>(v.data(), v.size());
}

} // namespace Mqtt

// ═══════════════════════════════════════════════════════════════════════════
// MqttBrokerApp
// Listens on TCP:1883, handles CONNECT → CONNACK, receives PUBLISH
// ═══════════════════════════════════════════════════════════════════════════

class MqttBrokerApp : public Application {
public:
  static TypeId GetTypeId() {
    static TypeId t = TypeId("MqttBrokerApp")
      .SetParent<Application>().SetGroupName("Tutorial")
      .AddConstructor<MqttBrokerApp>();
    return t;
  }
  void Setup(uint16_t port, const std::string &id) {
    m_port=port; m_id=id;
  }

private:
  void StartApplication() override {
    m_sock = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
    m_sock->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
    m_sock->Listen();
    m_sock->SetAcceptCallback(
      MakeNullCallback<bool,Ptr<Socket>,const Address&>(),
      MakeCallback(&MqttBrokerApp::OnAccept, this));
    NS_LOG_INFO(m_id << " listening on TCP:" << m_port);
  }
  void StopApplication() override {
    for(auto &s:m_clients) s->Close();
    m_clients.clear();
    if(m_sock){ m_sock->Close(); m_sock=nullptr; }
  }
  void OnAccept(Ptr<Socket> s, const Address&) {
    m_clients.push_back(s);
    s->SetRecvCallback(MakeCallback(&MqttBrokerApp::OnRecv, this));
  }
  void OnRecv(Ptr<Socket> s) {
    Ptr<Packet> p;
    while((p=s->Recv())) {
      if(p->GetSize()<2) continue;
      uint8_t b[2]; p->CopyData(b,2);
      uint8_t mtype=(b[0]>>4)&0x0F;
      if(mtype==1) // CONNECT → reply CONNACK
        s->Send(Mqtt::ToPkt(Mqtt::ConnAck()));
      // PUBLISH(3), SUBSCRIBE(8) → receive only, no reply (QoS 0)
    }
  }
  Ptr<Socket>              m_sock{nullptr};
  std::vector<Ptr<Socket>> m_clients;
  uint16_t                 m_port{MQTT_PORT};
  std::string              m_id;
};

// ═══════════════════════════════════════════════════════════════════════════
// MqttSensorApp
// Real MQTT CONNECT then PUBLISH loop
// Topics and payloads match real hardware PCAP
// ═══════════════════════════════════════════════════════════════════════════

class MqttSensorApp : public Application {
public:
  static TypeId GetTypeId() {
    static TypeId t = TypeId("MqttSensorApp")
      .SetParent<Application>().SetGroupName("Tutorial")
      .AddConstructor<MqttSensorApp>();
    return t;
  }
  void Setup(Ipv4Address broker, uint16_t port,
             const std::string &clientId,
             const std::string &topic,
             double intervalSec,
             double stopTime) {
    m_broker=broker; m_port=port; m_cid=clientId;
    m_topic=topic; m_interval=Seconds(intervalSec);
    m_stopAt=Seconds(stopTime);
  }

private:
  void StartApplication() override {
    m_sock = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
    m_sock->SetConnectCallback(
      MakeCallback(&MqttSensorApp::OnConn, this),
      MakeNullCallback<void,Ptr<Socket>>());
    m_sock->SetRecvCallback(MakeCallback(&MqttSensorApp::OnRecv, this));
    m_sock->Connect(InetSocketAddress(m_broker, m_port));
  }
  void StopApplication() override {
    m_ev.Cancel();
    if(m_sock){ m_sock->Close(); m_sock=nullptr; }
  }
  void OnConn(Ptr<Socket> s) {
    s->Send(Mqtt::ToPkt(Mqtt::Connect(m_cid)));
    Simulator::Schedule(MilliSeconds(150), &MqttSensorApp::DoPub, this);
  }
  void OnRecv(Ptr<Socket> s) {
    Ptr<Packet> p; while((p=s->Recv())){}
  }
  void DoPub() {
    if(!m_sock || Simulator::Now()>=m_stopAt) return;
    m_sock->Send(Mqtt::ToPkt(Mqtt::Publish(m_topic, BuildPayload())));
    m_cnt++;
    m_ev = Simulator::Schedule(m_interval, &MqttSensorApp::DoPub, this);
  }

  // Payloads match real hardware PCAP exactly
  std::string BuildPayload() {
    if(m_topic=="rpi2/broadcast")        return std::to_string(m_cnt+1);
    if(m_topic=="hazards/flame_1")       return "0";
    if(m_topic=="hazards/flame_alert")   return "Flame Detected!";
    if(m_topic=="hazards/water_level_1") return "No Echo";
    if(m_topic=="traffic/ultrasonic_1") {
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(3) << (18.0+(m_cnt%5)*0.5);
      return ss.str();
    }
    if(m_topic=="traffic/motion_1")
      return (m_cnt%2==0) ? "Motion Detected!" : "No Motion";
    return "data";
  }

  Ptr<Socket>  m_sock{nullptr};
  Ipv4Address  m_broker;
  uint16_t     m_port{MQTT_PORT};
  std::string  m_cid, m_topic;
  Time         m_interval, m_stopAt;
  EventId      m_ev;
  uint32_t     m_cnt{0};
};

// ═══════════════════════════════════════════════════════════════════════════
// MqttAttackApp  (the Raspberry Pi attacker)
//
// Key behaviour:
//  1. Spoofs a sensor IP for every attack connection
//     → cycles through g_spoof[] so traffic appears to come from ESP32s
//  2. Packet sizes are randomly sampled from Gaussian(mean, std) per mode
//     → not fixed to the mean — evades simple size-based detection
//  3. RL agent picks the active mode; SetMode() switches it live
//  4. Targets alternate between Broker0 and Broker1
// ═══════════════════════════════════════════════════════════════════════════

class MqttAttackApp : public Application {
public:
  static TypeId GetTypeId() {
    static TypeId t = TypeId("MqttAttackApp")
      .SetParent<Application>().SetGroupName("Tutorial")
      .AddConstructor<MqttAttackApp>();
    return t;
  }
  void Setup(Ipv4Address broker0, Ipv4Address broker1,
             uint16_t port, int mode, double stopTime) {
    m_b0=broker0; m_b1=broker1; m_port=port;
    m_mode=mode; m_stopAt=Seconds(stopTime);
  }
  void SetMode(int m)   { m_mode=m; }
  int  GetMode()  const { return m_mode; }
  int  GetTarget() const{ return m_targetIdx; }

private:
  void StartApplication() override { SchedNext(); }
  void StopApplication()  override {
    m_ev.Cancel();
    for(auto &s:m_socks) s->Close();
    m_socks.clear();
  }

  void SchedNext() {
    // Inter-packet time: Gaussian jitter around CTGAN mean
    double base = ATK[m_mode].deltaTime;
    Ptr<NormalRandomVariable> nrv = CreateObject<NormalRandomVariable>();
    nrv->SetAttribute("Mean",     DoubleValue(base));
    nrv->SetAttribute("Variance", DoubleValue(base*0.04)); // 20% std
    double d = std::max(0.01, nrv->GetValue());
    m_ev = Simulator::Schedule(Seconds(d), &MqttAttackApp::Fire, this);
  }

  // Random packet size from CTGAN Gaussian distribution
  uint32_t SamplePktSize() {
    Ptr<NormalRandomVariable> nrv = CreateObject<NormalRandomVariable>();
    nrv->SetAttribute("Mean",     DoubleValue(ATK[m_mode].pktMean));
    nrv->SetAttribute("Variance", DoubleValue(ATK[m_mode].pktStd * ATK[m_mode].pktStd));
    double sz = nrv->GetValue();
    // Clamp to realistic TCP packet range
    return (uint32_t)std::max(54.0, std::min(1460.0, sz));
  }

  // Pick next spoofed IP (cycle through 6 ESP32 slot-A addresses)
  Ipv4Address NextSpoofIp() {
    Ipv4Address ip = g_spoof[m_spoofIdx % 6];
    m_spoofIdx++;
    return ip;
  }

  void Fire() {
    if(Simulator::Now() >= m_stopAt) return;

    // Alternate target: even fires → Broker0, odd fires → Broker1
    Ipv4Address target = (m_fireCount % 2 == 0) ? m_b0 : m_b1;
    m_targetIdx = (m_fireCount % 2 == 0) ? 0 : 1;
    m_fireCount++;

    // Get random packet size from CTGAN distribution
    uint32_t pktSize = SamplePktSize();

    // Open new TCP socket — source IP = spoofed sensor IP
    Ptr<Socket> s = Socket::CreateSocket(GetNode(),
                                          TcpSocketFactory::GetTypeId());
    m_socks.push_back(s);

    // Bind to spoofed sensor IP (port 0 = OS picks ephemeral port)
    Ipv4Address spoofIp = NextSpoofIp();
    s->Bind(InetSocketAddress(spoofIp, 0));

    int mode = m_mode;
    uint32_t sz = pktSize;

    s->SetConnectCallback(
      MakeBoundCallback(&MqttAttackApp::OnConn, this, mode, sz),
      MakeNullCallback<void,Ptr<Socket>>());
    s->Connect(InetSocketAddress(target, m_port));

    SchedNext();
  }

  static void OnConn(MqttAttackApp *self, int mode, uint32_t sz,
                     Ptr<Socket> s) {
    std::vector<uint8_t> frame;
    std::string rnd = std::to_string(rand() % 65535);

    switch(mode) {
      case 0: // SYN TCP Flooding — minimal CONNECT, stress TCP stack
        frame = Mqtt::Connect("syn_" + rnd);
        break;
      case 1: // Basic Connect Flooding — standard CONNECT flood
        frame = Mqtt::Connect("flood_" + rnd);
        break;
      case 2: // Delayed Connect Flooding — connect then delay payload
        Simulator::Schedule(MilliSeconds(500), [s, rnd](){
          s->Send(Mqtt::ToPkt(Mqtt::Connect("delay_" + rnd)));
        });
        return;
      case 3: // Invalid Subscription Flooding — malformed SUBSCRIBE
        frame = Mqtt::Subscribe("invalid/##/bad_topic", rand()%65535);
        break;
      case 4: // Connect with WILL payload — largest packets
        frame = Mqtt::Connect("will_" + rnd, true,
                               "hazards/flame_alert",
                               "ALERT_" + std::to_string(rand()%1000));
        break;
      default:
        frame = Mqtt::Connect("atk_" + rnd);
        break;
    }

    // Pad or trim frame to match the CTGAN-sampled random size
    // This ensures packet size distribution matches the model
    if(frame.size() < sz) {
      // Pad with zero bytes (simulates extra payload entropy)
      std::vector<uint8_t> padded = frame;
      padded.resize(sz, 0x00);
      s->Send(Create<Packet>(padded.data(), padded.size()));
    } else {
      s->Send(Mqtt::ToPkt(frame));
    }
  }

  Ipv4Address              m_b0, m_b1;
  uint16_t                 m_port{MQTT_PORT};
  int                      m_mode{0};
  int                      m_targetIdx{0};
  uint32_t                 m_fireCount{0};
  uint32_t                 m_spoofIdx{0};
  Time                     m_stopAt;
  EventId                  m_ev;
  std::vector<Ptr<Socket>> m_socks;
};

// ═══════════════════════════════════════════════════════════════════════════
// RL ATTACKER STATE
// ═══════════════════════════════════════════════════════════════════════════

struct RLState {
  int    mode{0}, steps{0};
  double q[N_MODES], eps, mUsed[N_MODES], mRew[N_MODES];
  Ptr<MqttAttackApp> app;

  RLState() : eps(0.90) {
    for(int i=0;i<N_MODES;i++){q[i]=0;mUsed[i]=0;mRew[i]=0;}
  }

  int select() {
    Ptr<UniformRandomVariable> u = CreateObject<UniformRandomVariable>();
    if(u->GetValue(0,1) < eps) {
      return (int)u->GetValue(0, N_MODES);
    }
    int best=0;
    for(int i=1;i<N_MODES;i++) if(q[i]>q[best]) best=i;
    return best;
  }

  void update(int m, double r) {
    double bn=q[0];
    for(int i=1;i<N_MODES;i++) if(q[i]>bn) bn=q[i];
    q[m] += LR_RL * (r + GAMMA*bn - q[m]);
    mUsed[m]++; mRew[m]+=r;
    eps = std::max(EPS_FLOOR, eps*EPS_DECAY);
    steps++;
  }
};

static RLState g_rl;

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

void MkDir(const std::string &p) { mkdir(p.c_str(), 0755); }
std::string Out(const std::string &f) { return g_outDir+"/"+f; }

// Get loss ratio for attack flows from Pi IP
double GetLoss() {
  if(!g_mon||!g_cls) return 0.5;
  g_mon->CheckForLostPackets();
  auto st = g_mon->GetFlowStats();
  uint64_t tx=0, lost=0;
  for(auto &f:st) {
    auto t = g_cls->FindFlow(f.first);
    // Count flows going TO either broker on MQTT port
    if((t.destinationAddress==g_b0 || t.destinationAddress==g_b1)
        && t.destinationPort==MQTT_PORT) {
      // Only attacker flows — identified by source matching a spoofed IP
      for(int i=0;i<6;i++) {
        if(t.sourceAddress==g_spoof[i]) {
          tx   += f.second.txPackets;
          lost += f.second.lostPackets;
          break;
        }
      }
    }
  }
  return tx>0 ? (double)lost/tx : 0.5;
}

NetDeviceContainer WifiNet(NodeContainer nodes, const std::string &ssid,
                            YansWifiPhyHelper &phy) {
  WifiHelper w;
  w.SetStandard(WIFI_STANDARD_80211b);
  w.SetRemoteStationManager("ns3::ConstantRateWifiManager",
    "DataMode",    StringValue("DsssRate11Mbps"),
    "ControlMode", StringValue("DsssRate11Mbps"));
  WifiMacHelper mac;
  mac.SetType("ns3::AdhocWifiMac");
  return w.Install(phy, mac, nodes);
}

// ═══════════════════════════════════════════════════════════════════════════
// RL EVALUATION CALLBACK
// Fires every EVAL_INTERVAL seconds during the attack window
// ═══════════════════════════════════════════════════════════════════════════

void RLEval() {
  double now = Simulator::Now().GetSeconds();
  if(now >= g_simTime-1.0) return;

  double loss   = GetLoss();
  double reward = 1.0 - 2.0*loss;
  if(g_rl.mode==4) reward += 0.15; // WILL payload bonus (large packets)

  int prev = g_rl.mode;
  g_rl.update(prev, reward);
  int next = g_rl.select();

  if(next != prev && g_rl.app)
    g_rl.app->SetMode(next);
  g_rl.mode = next;

  // Log
  RLEntry e;
  e.t=now; e.step=g_rl.steps;
  e.prevMode=prev; e.newMode=next;
  e.loss=loss; e.reward=reward; e.eps=g_rl.eps;
  e.targetBroker = g_rl.app ? g_rl.app->GetTarget() : 0;
  for(int i=0;i<N_MODES;i++) e.q[i]=g_rl.q[i];
  g_rlLog.push_back(e);

  NS_LOG_INFO("[RL] t=" << std::fixed << std::setprecision(1) << now
    << "s  step=" << g_rl.steps
    << "  loss=" << std::setprecision(3) << loss
    << "  reward=" << std::setprecision(2) << reward
    << "  eps=" << std::setprecision(3) << g_rl.eps
    << "  " << ATK[prev].shortName << "->" << ATK[next].shortName
    << "  target=Broker" << e.targetBroker);

  Simulator::Schedule(Seconds(g_evalInterval), &RLEval);
}

// ─── Per-second timeseries snapshot ──────────────────────────────────────
void SnapTS() {
  if(!g_mon||!g_cls) return;
  g_mon->CheckForLostPackets();
  auto st = g_mon->GetFlowStats();
  uint64_t sB=0, aB=0;
  for(auto &f:st) {
    auto t = g_cls->FindFlow(f.first);
    bool isAtk=false;
    for(int i=0;i<6;i++)
      if(t.sourceAddress==g_spoof[i]){ isAtk=true; break; }
    if(isAtk) aB+=f.second.txBytes;
    else       sB+=f.second.txBytes;
  }
  g_ts.push_back({Simulator::Now().GetSeconds(), sB-g_prevSB, aB-g_prevAB});
  g_prevSB=sB; g_prevAB=aB;
  if(Simulator::Now().GetSeconds() < g_simTime-0.5)
    Simulator::Schedule(Seconds(1.0), &SnapTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// CSV WRITERS
// ═══════════════════════════════════════════════════════════════════════════

void WriteFlowsCsv() {
  std::ofstream f(Out("flows.csv"));
  f << "flow_id,src_ip,src_port,dst_ip,dst_port,"
    << "tx_packets,rx_packets,lost_packets,loss_pct,"
    << "avg_delay_ms,throughput_mbps,traffic_type\n";
  for(auto &fl:g_mon->GetFlowStats()) {
    auto t = g_cls->FindFlow(fl.first);
    double lp = fl.second.txPackets>0
      ? 100.0*fl.second.lostPackets/fl.second.txPackets : 0;
    double ad = fl.second.rxPackets>0
      ? fl.second.delaySum.GetSeconds()/fl.second.rxPackets*1000 : 0;
    double tp = 0;
    if(fl.second.timeLastRxPacket>fl.second.timeFirstTxPacket) {
      double d=(fl.second.timeLastRxPacket
               -fl.second.timeFirstTxPacket).GetSeconds();
      tp=fl.second.rxBytes*8.0/d/1e6;
    }
    bool isAtk=false;
    for(int i=0;i<6;i++)
      if(t.sourceAddress==g_spoof[i]){ isAtk=true; break; }
    f << fl.first<<","<<t.sourceAddress<<","<<t.sourcePort<<","
      <<t.destinationAddress<<","<<t.destinationPort<<","
      <<fl.second.txPackets<<","<<fl.second.rxPackets<<","
      <<fl.second.lostPackets<<","
      <<std::fixed<<std::setprecision(2)<<lp<<","
      <<std::setprecision(3)<<ad<<","<<std::setprecision(6)<<tp<<","
      <<(isAtk?"attack":"sensor")<<"\n";
  }
  NS_LOG_INFO("Written: " << Out("flows.csv"));
}

void WriteRLCsv() {
  std::ofstream f(Out("rl_log.csv"));
  f<<"time_s,step,prev_mode,prev_name,new_mode,new_name,"
   <<"loss_ratio,reward,epsilon,target_broker,"
   <<"q0_SYN,q1_BASIC,q2_DELAY,q3_INVSUB,q4_WILL\n";
  for(auto &e:g_rlLog) {
    f<<std::fixed<<std::setprecision(2)<<e.t<<","<<e.step<<","
     <<e.prevMode<<","<<ATK[e.prevMode].shortName<<","
     <<e.newMode <<","<<ATK[e.newMode].shortName<<","
     <<std::setprecision(4)<<e.loss<<","<<e.reward<<","<<e.eps<<","
     <<"Broker"<<e.targetBroker<<","
     <<e.q[0]<<","<<e.q[1]<<","<<e.q[2]<<","<<e.q[3]<<","<<e.q[4]<<"\n";
  }
  NS_LOG_INFO("Written: " << Out("rl_log.csv"));
}

void WriteTsCsv() {
  std::ofstream f(Out("timeseries.csv"));
  f<<"time_s,sensor_bytes,attack_bytes,sensor_kbps,attack_kbps\n";
  for(auto &e:g_ts) {
    f<<std::fixed<<std::setprecision(1)<<e.t<<","
     <<e.sBytes<<","<<e.aBytes<<","
     <<std::setprecision(2)<<e.sBytes*8.0/1000<<","
     <<e.aBytes*8.0/1000<<"\n";
  }
  NS_LOG_INFO("Written: " << Out("timeseries.csv"));
}

void WriteSumCsv() {
  auto st = g_mon->GetFlowStats();
  uint64_t sTx=0,sRx=0,sL=0,sTB=0,aTx=0,aRx=0,aL=0,aTB=0;
  for(auto &fl:st) {
    auto t=g_cls->FindFlow(fl.first);
    bool isAtk=false;
    for(int i=0;i<6;i++)
      if(t.sourceAddress==g_spoof[i]){ isAtk=true; break; }
    if(isAtk){ aTx+=fl.second.txPackets; aRx+=fl.second.rxPackets;
               aL+=fl.second.lostPackets; aTB+=fl.second.txBytes; }
    else      { sTx+=fl.second.txPackets; sRx+=fl.second.rxPackets;
               sL+=fl.second.lostPackets; sTB+=fl.second.txBytes; }
  }
  auto pct=[](uint64_t l,uint64_t t)->double{
    return t>0?100.0*l/t:0;};
  std::ofstream f(Out("summary.csv"));
  f<<"category,tx_packets,rx_packets,lost_packets,loss_pct,tx_bytes\n";
  f<<"sensor,"<<sTx<<","<<sRx<<","<<sL<<","
   <<std::fixed<<std::setprecision(2)<<pct(sL,sTx)<<","<<sTB<<"\n";
  f<<"attack,"<<aTx<<","<<aRx<<","<<aL<<","
   <<pct(aL,aTx)<<","<<aTB<<"\n";
  f<<"\nattacker_rl_final_state\n";
  f<<"steps,final_mode,final_mode_name,final_epsilon,"
   <<"q0_SYN,q1_BASIC,q2_DELAY,q3_INVSUB,q4_WILL\n";
  f<<g_rl.steps<<","<<g_rl.mode<<","<<ATK[g_rl.mode].shortName<<","
   <<std::setprecision(4)<<g_rl.eps<<","
   <<g_rl.q[0]<<","<<g_rl.q[1]<<","
   <<g_rl.q[2]<<","<<g_rl.q[3]<<","<<g_rl.q[4]<<"\n";
  NS_LOG_INFO("Written: " << Out("summary.csv"));
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char *argv[])
{
  // ── Command-line parameters ────────────────────────────────────────────
  CommandLine cmd(__FILE__);
  cmd.AddValue("simTime",     "Simulation time (s)",       g_simTime);
  cmd.AddValue("attackStart", "Attack start time (s)",     g_attackStart);
  cmd.AddValue("evalInterval","RL eval interval (s)",      g_evalInterval);
  cmd.AddValue("epsilon",     "Initial RL epsilon",        g_epsilon);
  cmd.AddValue("seed",        "Random seed",               g_seed);
  cmd.AddValue("outDir",      "Output directory",          g_outDir);
  cmd.AddValue("broker0Ip",   "Broker0 IP address",        g_broker0Ip);
  cmd.AddValue("broker1Ip",   "Broker1 IP address",        g_broker1Ip);
  cmd.AddValue("piIp",        "Raspberry Pi IP address",   g_piIp);
  cmd.Parse(argc, argv);

  // Apply epsilon from command line to RL state
  g_rl.eps = g_epsilon;

  Time::SetResolution(Time::NS);
  RngSeedManager::SetSeed(g_seed);
  LogComponentEnable("MqttDDoSRLSimulation", LOG_LEVEL_INFO);

  MkDir(g_outDir);
  MkDir(g_outDir + "/pcap");

  // ── 1. Create nodes ────────────────────────────────────────────────────
  // One network: 2 brokers + 12 sensors + 1 Pi attacker = 15 nodes
  NodeContainer brokers;   brokers.Create(2);    // Broker0, Broker1
  NodeContainer sensors;   sensors.Create(12);   // Sensor0..11
  NodeContainer piNode;    piNode.Create(1);      // Raspberry Pi

  // All nodes in one container for WiFi installation
  NodeContainer allNodes;
  allNodes.Add(brokers);
  allNodes.Add(sensors);
  allNodes.Add(piNode);

  // ── 2. Mobility — grid layout matching physical setup ──────────────────
  MobilityHelper mob;
  mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();

  // Brokers — centre
  pos->Add(Vector(50.0, 50.0, 0));   // Broker0
  pos->Add(Vector(50.0, 100.0, 0));  // Broker1

  // Sensors 0-5 (left side → Broker0), 2 per row = 3 rows of ESP32
  pos->Add(Vector(10.0, 30.0, 0));   // Sensor0  ESP32 #1
  pos->Add(Vector(10.0, 45.0, 0));   // Sensor1  ESP32 #1
  pos->Add(Vector(10.0, 60.0, 0));   // Sensor2  ESP32 #2
  pos->Add(Vector(10.0, 75.0, 0));   // Sensor3  ESP32 #2
  pos->Add(Vector(10.0, 90.0, 0));   // Sensor4  ESP32 #3
  pos->Add(Vector(10.0, 105.0, 0));  // Sensor5  ESP32 #3

  // Sensors 6-11 (right side → Broker1), 3 rows of ESP32
  pos->Add(Vector(90.0, 30.0, 0));   // Sensor6  ESP32 #4
  pos->Add(Vector(90.0, 45.0, 0));   // Sensor7  ESP32 #4
  pos->Add(Vector(90.0, 60.0, 0));   // Sensor8  ESP32 #5
  pos->Add(Vector(90.0, 75.0, 0));   // Sensor9  ESP32 #5
  pos->Add(Vector(90.0, 90.0, 0));   // Sensor10 ESP32 #6
  pos->Add(Vector(90.0, 105.0, 0));  // Sensor11 ESP32 #6

  // Raspberry Pi — bottom centre
  pos->Add(Vector(50.0, 140.0, 0));  // Pi (attacker)

  mob.SetPositionAllocator(pos);
  mob.Install(allNodes);

  // ── 3. Single WiFi network ─────────────────────────────────────────────
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());

  NetDeviceContainer devices = WifiNet(allNodes, "iot-network", phy);

  // ── 4. PCAP capture ───────────────────────────────────────────────────
  // Broker0 = node 0, Broker1 = node 1, Pi = node 14 (last)
  phy.EnablePcap(g_outDir+"/pcap/broker0",   devices.Get(0),  true);
  phy.EnablePcap(g_outDir+"/pcap/broker1",   devices.Get(1),  true);
  phy.EnablePcap(g_outDir+"/pcap/attacker",  devices.Get(14), true);

  // ── 5. Internet stack + IP addressing ─────────────────────────────────
  InternetStackHelper stack;
  stack.Install(allNodes);

  // Assign IPs matching real hardware addresses
  // We use a manual assignment to preserve the real IP addresses
  Ipv4AddressHelper ipv4;
  ipv4.SetBase("192.168.0.0", "255.255.255.0", "0.0.0.130");
  // Note: we assign manually node by node to get exact IPs

  // Manual IP assignment to preserve real hardware IPs
  // Node order: Broker0, Broker1, S0..S11, Pi
  std::string nodeIps[15] = {
    g_broker0Ip,       // Broker0  node 0
    g_broker1Ip,       // Broker1  node 1
    g_sensorIps[0],    // Sensor0  node 2
    g_sensorIps[1],    // Sensor1  node 3
    g_sensorIps[2],    // Sensor2  node 4
    g_sensorIps[3],    // Sensor3  node 5
    g_sensorIps[4],    // Sensor4  node 6
    g_sensorIps[5],    // Sensor5  node 7
    g_sensorIps[6],    // Sensor6  node 8
    g_sensorIps[7],    // Sensor7  node 9
    g_sensorIps[8],    // Sensor8  node 10
    g_sensorIps[9],    // Sensor9  node 11
    g_sensorIps[10],   // Sensor10 node 12
    g_sensorIps[11],   // Sensor11 node 13
    g_piIp,            // Pi       node 14
  };

  for(int i=0; i<15; i++) {
    Ptr<Node> node = allNodes.Get(i);
    Ptr<Ipv4> ipv4obj = node->GetObject<Ipv4>();
    int32_t ifIdx = ipv4obj->GetInterfaceForDevice(devices.Get(i));
    if(ifIdx < 0) {
      ifIdx = ipv4obj->AddInterface(devices.Get(i));
    }
    Ipv4InterfaceAddress addr(
      Ipv4Address(nodeIps[i].c_str()),
      Ipv4Mask("255.255.255.0"));
    ipv4obj->AddAddress(ifIdx, addr);
    ipv4obj->SetMetric(ifIdx, 1);
    ipv4obj->SetUp(ifIdx);
  }

  // Store resolved addresses
  g_b0 = Ipv4Address(g_broker0Ip.c_str());
  g_b1 = Ipv4Address(g_broker1Ip.c_str());
  g_pi = Ipv4Address(g_piIp.c_str());
  for(int i=0;i<12;i++)
    g_sAddr[i] = Ipv4Address(g_sensorIps[i].c_str());
  for(int i=0;i<6;i++)
    g_spoof[i] = Ipv4Address(g_spoofIps[i].c_str());

  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  NS_LOG_INFO("Network: 192.168.0.0/24  (single WiFi — mirrors real hardware)");
  NS_LOG_INFO("Broker0 : " << g_b0 << "  Broker1: " << g_b1);
  NS_LOG_INFO("Pi (atk): " << g_pi);
  NS_LOG_INFO("Spoofed IPs: " << g_spoof[0] << " " << g_spoof[1]
              << " " << g_spoof[2] << " " << g_spoof[3]
              << " " << g_spoof[4] << " " << g_spoof[5]);

  // ── 6. Broker applications ─────────────────────────────────────────────
  for(int b=0; b<2; b++) {
    Ptr<MqttBrokerApp> a = CreateObject<MqttBrokerApp>();
    a->Setup(MQTT_PORT, "broker"+std::to_string(b));
    brokers.Get(b)->AddApplication(a);
    a->SetStartTime(Seconds(0.5));
    a->SetStopTime(Seconds(g_simTime));
  }

  // ── 7. Sensor applications ─────────────────────────────────────────────
  // 12 sensors, 6 per broker
  // Topics match real PCAP — each ESP32 pair publishes different topics
  // ESP32 pairing: sensors 0+1, 2+3, 4+5 → Broker0
  //                sensors 6+7, 8+9, 10+11 → Broker1

  struct SensorCfg {
    int       idx;
    Ipv4Address *broker;
    const char  *topic;
    double       interval;
  };

  SensorCfg scfg[] = {
    // ESP32 #1 → Broker0
    {  0, &g_b0, "rpi2/broadcast",        2.0 },
    {  1, &g_b0, "hazards/flame_1",       4.0 },
    // ESP32 #2 → Broker0
    {  2, &g_b0, "hazards/flame_alert",   4.0 },
    {  3, &g_b0, "hazards/water_level_1", 4.0 },
    // ESP32 #3 → Broker0
    {  4, &g_b0, "traffic/ultrasonic_1",  1.0 },
    {  5, &g_b0, "traffic/motion_1",      1.0 },
    // ESP32 #4 → Broker1
    {  6, &g_b1, "rpi2/broadcast",        2.0 },
    {  7, &g_b1, "hazards/flame_1",       4.0 },
    // ESP32 #5 → Broker1
    {  8, &g_b1, "hazards/flame_alert",   4.0 },
    {  9, &g_b1, "hazards/water_level_1", 4.0 },
    // ESP32 #6 → Broker1
    { 10, &g_b1, "traffic/ultrasonic_1",  1.0 },
    { 11, &g_b1, "traffic/motion_1",      1.0 },
  };

  for(auto &c : scfg) {
    Ptr<MqttSensorApp> a = CreateObject<MqttSensorApp>();
    a->Setup(*(c.broker), MQTT_PORT,
             "Sensor_"+std::to_string(c.idx),
             c.topic, c.interval, g_simTime);
    sensors.Get(c.idx)->AddApplication(a);
    a->SetStartTime(Seconds(1.0 + c.idx*0.05)); // stagger starts
    a->SetStopTime(Seconds(g_simTime));
  }

  // ── 8. Attacker application (Raspberry Pi) ─────────────────────────────
  g_rl.mode = 0; // Start with SYN flood
  Ptr<MqttAttackApp> atkApp = CreateObject<MqttAttackApp>();
  atkApp->Setup(g_b0, g_b1, MQTT_PORT, g_rl.mode, g_simTime);
  piNode.Get(0)->AddApplication(atkApp);
  atkApp->SetStartTime(Seconds(g_attackStart));
  atkApp->SetStopTime(Seconds(g_simTime));
  g_rl.app = atkApp;

  // ── 9. FlowMonitor ─────────────────────────────────────────────────────
  FlowMonitorHelper fmh;
  g_mon = fmh.InstallAll();
  g_cls = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());

  // ── 10. RL evaluation schedule ─────────────────────────────────────────
  Simulator::Schedule(Seconds(g_attackStart + g_evalInterval), &RLEval);

  // ── 11. Timeseries snapshots ────────────────────────────────────────────
  Simulator::Schedule(Seconds(1.0), &SnapTS);

  // ── 12. NetAnim ─────────────────────────────────────────────────────────
  AnimationInterface anim(Out("mqtt-rl-anim.xml"));

  // Positions already set via SetConstantPosition above (via mobility)
  // Manually set for clarity
  anim.SetConstantPosition(brokers.Get(0),  50.0,  50.0);
  anim.SetConstantPosition(brokers.Get(1),  50.0, 100.0);
  for(int i=0;i<6;i++)
    anim.SetConstantPosition(sensors.Get(i),   10.0, 30.0+i*15.0);
  for(int i=6;i<12;i++)
    anim.SetConstantPosition(sensors.Get(i),   90.0, 30.0+(i-6)*15.0);
  anim.SetConstantPosition(piNode.Get(0), 50.0, 145.0);

  // Labels
  anim.UpdateNodeDescription(brokers.Get(0),
    "Broker0 ("+g_broker0Ip+")");
  anim.UpdateNodeDescription(brokers.Get(1),
    "Broker1 ("+g_broker1Ip+")");
  anim.UpdateNodeDescription(piNode.Get(0),
    "Pi-Attacker ("+g_piIp+") [RL]");

  const char *esp32Labels[6] = {
    "ESP32#1","ESP32#2","ESP32#3","ESP32#4","ESP32#5","ESP32#6"
  };
  for(int i=0;i<12;i++) {
    std::string lbl = std::string(esp32Labels[i/2])
                    + " S" + std::to_string(i)
                    + " (" + g_sensorIps[i] + ")";
    anim.UpdateNodeDescription(sensors.Get(i), lbl);
  }

  // Colors — matching old code style
  anim.UpdateNodeColor(brokers.Get(0),  34, 204, 136); // green
  anim.UpdateNodeColor(brokers.Get(1),  34, 204, 136); // green
  for(int i=0;i<12;i++)
    anim.UpdateNodeColor(sensors.Get(i), 68, 136, 255); // blue
  anim.UpdateNodeColor(piNode.Get(0), 255, 68, 68);     // red

  // Sizes
  anim.UpdateNodeSize(brokers.Get(0)->GetId(), 4.0, 4.0);
  anim.UpdateNodeSize(brokers.Get(1)->GetId(), 4.0, 4.0);
  anim.UpdateNodeSize(piNode.Get(0)->GetId(),  4.0, 4.0);
  for(int i=0;i<12;i++)
    anim.UpdateNodeSize(sensors.Get(i)->GetId(), 3.0, 3.0);

  anim.EnablePacketMetadata(true);

  // ── 13. Run ────────────────────────────────────────────────────────────
  Simulator::Stop(Seconds(g_simTime + 1.0));

  NS_LOG_INFO("========================================");
  NS_LOG_INFO(" MQTT DDoS RL Simulation");
  NS_LOG_INFO(" Single WiFi network — 192.168.0.0/24");
  NS_LOG_INFO(" 2 Brokers + 12 Sensors + 1 Pi attacker");
  NS_LOG_INFO(" Attack start: t=" << g_attackStart << "s");
  NS_LOG_INFO(" Spoofing: " << 6 << " ESP32 sensor IPs");
  NS_LOG_INFO(" Packet sizes: random (CTGAN Gaussian dist)");
  NS_LOG_INFO(" Output → " << g_outDir << "/");
  NS_LOG_INFO("========================================");

  Simulator::Run();

  // ── 14. Write outputs ──────────────────────────────────────────────────
  g_mon->CheckForLostPackets();
  g_mon->SerializeToXmlFile(Out("mqtt-rl-flowmonitor.xml"), true, true);
  WriteFlowsCsv();
  WriteRLCsv();
  WriteTsCsv();
  WriteSumCsv();

  // ── 15. Console summary ────────────────────────────────────────────────
  std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║  MQTT DDoS RL Simulation — Complete                     ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
  std::cout << "  Network     : 192.168.0.0/24 (single WiFi)\n";
  std::cout << "  Brokers     : " << g_broker0Ip << "  " << g_broker1Ip << "\n";
  std::cout << "  Pi attacker : " << g_piIp << "\n";
  std::cout << "  Spoofed IPs : ESP32 sensor addresses\n";
  std::cout << "  Pkt sizes   : random Gaussian (CTGAN dist)\n\n";
  std::cout << "  Output → " << g_outDir << "/\n";
  std::cout << "    pcap/broker0-*.pcap   (filter: mqtt in Wireshark)\n";
  std::cout << "    pcap/broker1-*.pcap\n";
  std::cout << "    pcap/attacker-*.pcap\n";
  std::cout << "    flows.csv\n";
  std::cout << "    rl_log.csv\n";
  std::cout << "    timeseries.csv\n";
  std::cout << "    summary.csv\n\n";

  std::cout << "  RL final state:  steps=" << g_rl.steps
            << "  eps=" << std::fixed << std::setprecision(3) << g_rl.eps
            << "  best_mode=" << ATK[g_rl.mode].shortName << "\n";
  std::cout << "  Q-table:\n";
  for(int m=0;m<N_MODES;m++)
    std::cout << "    [" << m << "] " << std::setw(7) << ATK[m].shortName
              << "  Q=" << std::setprecision(3) << g_rl.q[m]
              << "  used=" << (int)g_rl.mUsed[m] << "\n";

  Simulator::Destroy();
  NS_LOG_INFO("Done.");
  return 0;
}
