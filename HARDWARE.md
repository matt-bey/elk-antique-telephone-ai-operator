# Hardware — Antique Telephone AI Operator

## Phase 1: Modern Testing Hardware

Components used to validate software logic before integrating with the antique telephone.

| Component | Details |
|---|---|
| Raspberry Pi 5 (8GB) | Main compute board |
| USB microphone | Generic USB — development STT input |
| Speaker / portable Bluetooth speaker | Development audio output |
| Push buttons (×2) | Simulate crank trigger and hook switch (any 37-in-1 starter kit) |

## Phase 2: Antique Hardware Integration

Replaces the modern stand-ins above. The telephone is a **Kellogg Chicago antique wood wall telephone (early 1900s)**.

### Original Telephone Components

| Component | Details | Verification |
|---|---|---|
| Antique wood wall telephone (Kellogg Chicago) | Complete unit — earpiece, carbon microphone, crank, hook switch, ringer | ✅ Inspected, components catalogued |
| Earpiece | DC resistance: **63Ω** — coil intact, functional | ✅ Confirmed working |
| Carbon microphone | DC resistance: **~270Ω** in correct wall-mount orientation (reads ~2kΩ inverted — granules are mobile and responsive to gravity) | ✅ Confirmed working |
| Magneto crank | AC output — multimeter reads ~5V RMS but this is unreliable at 20Hz (meter calibrated for 50/60Hz); actual output is ~75–100V based on bell strike. **Output wires will be disconnected from circuit.** Internal contact plates act as a dry momentary switch when crank is engaged — used for GPIO trigger instead. | ✅ Confirmed — output disconnected, contacts used for GPIO |
| Hook switch | On/off-hook detection — **open on-hook, closed off-hook**. Wire as GPIO via 10KΩ pull-up to 3.3V: Pi reads HIGH on-hook (earpiece hung), LOW off-hook (earpiece lifted). Software debouncing required (leaf spring chatters on transition). | ✅ Confirmed — polarity verified |
| Mechanical ringer | Two 800Ω coils in series — **total impedance: 1604Ω**. Three terminals: outer two connect to coils; center terminal is unused center tap between coils (ignore for this project). Clapper responds to crank AC signal — mechanically intact. Requires 90V AC at 20Hz for full bell strike. | ✅ Measured — working |

#### Carbon Microphone Notes

Resistance is orientation-sensitive — must be tested and mounted in correct wall-mount orientation or granules shift and resistance climbs significantly. Tap the element gently if it has been handled out of orientation before measuring. Granules confirmed mobile. Functional audio test confirmed: audible signal at 5V bias through earpiece with no amplification; signal strength increased meaningfully when series resistor dropped from 1KΩ to 220Ω, confirming element responds correctly to bias current.

**Confirmed bias + attenuator circuit (bench-validated on Pi 5 with 12V supply):** 330Ω + 100Ω in series (430Ω total) from 12V to mic(+); mic(−) to GND. Static resistance reads ~250–270Ω in correct orientation, but effective resistance under operating bias current drops to ~53Ω — Node A sits at ~1.4V DC, not the ~4.5V the static measurement would predict. This is normal non-linear behavior of carbon granules under current. AC signal tapped at Node A via 10µF electrolytic cap (+ leg toward Node A), fed through a **5.1kΩ series / 1kΩ shunt** voltage divider into the Vantec USB DAC mic-in. No active preamp required. 10kΩ/1kΩ (original estimate) was too much attenuation; no resistor (cap directly to mic-in) clipped and distorted. 5.1kΩ/1kΩ confirmed clean signal at 100% ALSA capture level.

#### Ringer Notes

The clapper visibly moves when driven by the magneto crank (~3mA at 5V) but does not reach the bells — this is a power issue, not mechanical. At 90V/20Hz the coils will see ~56mA, providing sufficient force for full bell strikes. Do not select a ring generator module rated only for standard 600Ω POTS ringers — this ringer is 1604Ω and requires a module specified to deliver adequate current into that load.

### Additional Circuits Required (Planned)

| Component | Details | Source |
|---|---|---|
| USB audio adapter (Vantec NBA-120U) | Primary audio I/O. 3.5mm headphone-out drives the 63Ω earpiece directly at audible volume (bench-verified on Pi Zero 2W — no amplifier needed). 3.5mm mic-in accepts carbon mic signal via bias + attenuator. Plug-and-play ALSA device. **On Pi 5: card 2, device 0. Use `plughw:2,0` for mono playback compatibility. ALSA output level ~84 for good earpiece volume; capture level 100.** | Vantec |
| Carbon microphone bias + attenuator | 12V bias: 330Ω + 100Ω in series → mic(+); mic(−) → GND. AC signal tapped at Node A via 10µF electrolytic cap → **5.1kΩ series / 1kΩ shunt** divider → USB DAC mic-in. Bench-validated: clean signal at 100% capture level. | |
| Crank signal conditioning | Magneto output disconnected. Internal crank contact plates (dry momentary switch) wired to GPIO via 10KΩ pull-up to 3.3V. Software debouncing required; wait for crank to fully stop before triggering AI operator. | |
| Hook switch signal conditioning | Hook switch wired to GPIO via 10KΩ pull-up to 3.3V. Pi reads HIGH on-hook, LOW off-hook. Software debouncing required (leaf spring chatters on transition). | |
| Relay driver circuit | 2N3904 + 1N4007 + 5V relay — switches ring generator output to ringer coil. RC snubber across relay contacts required: 150Ω + 0.1µF/250V film cap — protects against inductive spike from 1604Ω ringer coil when relay opens. | |
| Ring generator module | CEL Black Magic LSN 12-86 (part LS128620), 12V DC in → 86 Vrms sine wave at 20Hz. Into 1604Ω: ~54mA, ~4.6W — within 5W intermittent rating. ±4% load regulation. Idle: ~95mA. | Cambridge Electronics Labs (also on eBay) |

#### Ring Generator Candidate Modules

| Module | Input | Output | Notes | Status |
|---|---|---|---|---|
| **CEL Black Magic LSN 12-86** | 12V DC | **86 Vrms / 20Hz sine** | Confirmed available eBay. Part LS128620 (20Hz variant). ~$31. | **✅ Selected** |
| Sandman DSI9P Ring Voltage Booster | 12V DC | 90V AC | Standalone use unconfirmed — requires call to verify | Backup only |
| PowerDsine PCR-SIN03V12F20-C | 12V DC | 70V AC / 20Hz | Lower voltage, likely insufficient for 1604Ω load | Ruled out |
| Model Railroad Control Systems | 12V DC | 70V AC / 20Hz | Lower voltage, likely insufficient for 1604Ω load | Ruled out |

#### CEL Black Magic LSN 12-86 — Confirmed Specs

Cambridge Electronics Labs (Somerville MA) manufactures the **Black Magic Telephone Ringing Generator**. Manufacturer contacted directly — helpful and responsive.

| Spec | Value |
|---|---|
| Part number | LS128620 (12V in, 86V out, 20Hz) |
| Input voltage | 12V DC (11–13V operating range) |
| Output voltage | 86 Vrms ± 4% |
| Output frequency | 20 Hz ± 0.5 Hz |
| Output power | 5W intermittent |
| Idle current | ~95mA @ 12V |
| Load into 1604Ω | ~54mA, ~4.6W — within rating |
| Size | 38.1 × 38.1 × 12.7mm |
| Pinout | + − input (1.0" spacing); ~ ~ output (0.6" spacing); 0.025" square pins |
| Mounting | Direct solder-down to PCB or Mill-Max clips #1 or #47 |

**Why sine wave over square wave:** Tighter load regulation (±4% vs ±15% for square wave model) ensures consistent current to the ringer near the 5W limit. No harmonics means no buzzing or erratic clapper behavior in the resonant mechanical system. No wave-shaping filter required.

**Important notes from datasheet:**
- Output is **not short-circuit protected** — requires ≥300Ω series resistance in circuit. The 1604Ω ringer satisfies this; no additional resistance needed.
- **Input is not reverse-polarity protected** — verify polarity before powering.
- Highly inductive loads (the mechanical ringer qualifies) may require conditioning — the RC snubber on the relay contacts addresses this.

**References:**
- Datasheet: `datasheets/black-magic-telephone-ringing-generator-sine-lsn-model.pdf`
- Design guide: `datasheets/black-magic-telecom-design-tricks.pdf`
- Contact: +1 617 629-2805 / camblab@attglobal.net / 20 Chester Street, Somerville MA 02144

### LM386 Amplifier (Held in Reserve)

The Vantec USB DAC drives the 63Ω earpiece directly at audible volume on the bench — no active amplifier is required for the primary build. The LM386 is retained as a fallback if the final install reveals insufficient volume (e.g., earpiece positioning in the mounted cabinet, enclosure damping, or user preference). Build-ready BOM and pinout are kept below so promotion from reserve requires no re-research.

| Component | Value / Details |
|---|---|
| LM386 Audio Amplifier IC | 8-pin DIP |
| Electrolytic capacitor C1 | 10µF — input coupling |
| Electrolytic capacitor C2 | 220µF — output coupling |
| Ceramic capacitor C3 | 0.1µF — power supply decoupling |
| Potentiometer | 10kΩ — volume control |
| 8-pin IC socket | — |

**LM386 key specs:** default gain 20× (26dB), up to 200× with modification; drives 63Ω earpiece comfortably at 5V supply.

**LM386 pin wiring (planned):**
- Pin 6 → +5V (Pi GPIO)
- Pin 4 → GND
- Pin 3 → Audio input (through C1 from Pi audio out)
- Pin 2 → Volume potentiometer wiper
- Pin 5 → Audio output (through C2 to earpiece)

### Crank Signal Conditioning Circuit (Planned)

The magneto output is disconnected from the telephone circuit. The crank's internal contact plates act as a dry momentary switch that closes when the crank handle is pressed in and engaged.

```
3.3V ──── 10KΩ ──┬──── Pi GPIO (reads HIGH at rest, LOW when crank contacts close)
                 │
           [crank contacts]
                 │
                GND
```

Software must debounce the contact and wait for crank rotation to fully stop before triggering the AI operator state machine. All parts on hand.

### Hook Switch Signal Conditioning Circuit (Planned)

The hook switch is open on-hook (earpiece hung on the cradle) and closed off-hook (earpiece lifted).

```
3.3V ──── 10KΩ ──┬──── Pi GPIO (reads HIGH on-hook, LOW off-hook)
                 │
           [hook switch]
                 │
                GND
```

Software must debounce the contact — the leaf spring chatters briefly when the earpiece is lifted or placed on the cradle. All parts on hand.

## Hardware Integration Test — Wiring Guide

Bench-validated wiring for the outbound call test: momentary button simulates the crank, real earpiece on the DAC headphone output, real carbon mic through the bias + attenuator circuit into the DAC mic input.

### Circuit 1: Crank / Button Simulation

No external pull-up resistor needed — use the Pi's internal pull-up (`GPIO.PUD_UP` in software). Pi reads HIGH at rest, LOW when button pressed.

| Wire | From | To |
|---|---|---|
| 1 | Pi GPIO pin (crank) | One leg of momentary button |
| 2 | Other leg of momentary button | GND |

### Circuit 2: Earpiece (Audio Output)

Direct connection — no additional components.

| Wire | From | To |
|---|---|---|
| 1 | 3.5mm plug Tip | One earpiece terminal |
| 2 | 3.5mm plug Sleeve | Other earpiece terminal |

Plug into Vantec DAC headphone output. Set ALSA output level to ~84.

### Circuit 3: Carbon Microphone Bias + Signal Extraction

Two current paths share **Node A** — the junction of the bias resistors and the mic(+) terminal.

**DC bias path** (drives current through the element):

| Wire | From | To |
|---|---|---|
| 1 | 12V (+) | One end of 330Ω resistor |
| 2 | Other end of 330Ω | One end of 100Ω resistor |
| 3 | Other end of 100Ω | **Node A** |
| 4 | **Node A** | mic (+) terminal |
| 5 | mic (−) terminal | GND |

**AC signal path** (extracts audio from Node A to DAC):

| Wire | From | To |
|---|---|---|
| 6 | **Node A** (same physical point as wires 3 and 4) | (+) leg of 10µF electrolytic cap |
| 7 | (−) leg of 10µF cap | One end of 5.1kΩ resistor |
| 8 | Other end of 5.1kΩ resistor | **Node B** |
| 9 | **Node B** | 3.5mm plug Tip (mic-in jack) |
| 10 | **Node B** (same physical point as wire 9) | One end of 1kΩ resistor |
| 11 | Other end of 1kΩ resistor | GND |
| 12 | 3.5mm plug Sleeve (mic-in jack) | GND |

**Capacitor orientation:** electrolytic — (+) long leg toward Node A (higher DC potential), (−) short leg toward the 5.1kΩ resistor.

**Common GND:** 12V supply (−), mic (−), 1kΩ bottom leg, 3.5mm mic-in sleeve, and Pi GND pin must all connect to the same GND point. Required for the signal path to have a common reference.

**Power during testing:** USB-C brick for the Pi, 12V Mean Well for the mic bias. Run one wire from a Pi GND pin to the 12V supply (−) to tie the grounds.

## Power Architecture (Phase 2)

### Mains Supply Components

| Component | Details | Source |
|---|---|---|
| Mains inlet cable | 3-prong extension cord (cut) — L/N/E feed into box | |
| Inline fuse holder + fuses | BOJACK inline screw type, 5×20mm, 250V, 16 AWG leads — 1A fuse; splices into L wire | BOJACK |
| Mean Well IRM-30-5 (or equivalent) | PCB-mount, 5V/6A, mains AC in — powers Pi 5 via GPIO 5V pins | Mean Well |
| Mean Well IRM-30-12 (or equivalent) | PCB-mount, 12V/2.5A, mains AC in — powers ring generator + carbon mic bias | Mean Well |
| Wire nuts | Join +12V rail wires and common GND wires — one nut per rail | |

**Mains wiring:** L and N daisy-chained between both PSU input terminals using stranded silicone wire (on hand). Earth connected to both PSU E terminals. Fuse on L wire before first PSU.

```
Extension cord (L, N, E)
        │
   [Inline fuse — L wire only]
        │
        ├──→ IRM-30-5  L ──┐  N ──┐  E
        │                  │      │
        └──→ IRM-30-12  L ←┘  N ←┘  E
```

### DC Distribution

| Supply | Voltage | Load |
|---|---|---|
| IRM-30-5 | 5V / 5A | Raspberry Pi 5 (USB-C) |
| IRM-30-12 | 12V / 2.5A | Ring generator module, carbon mic bias circuit |

**Common ground:** Pi GND GPIO pin joins the GND wire nut, tying both supply grounds together. Required for relay driver and signal circuits to share a reference.

**12V supply eliminates the DC-DC boost converter** previously planned — the 12V rail provides the bias voltage for the carbon microphone directly.

### Rail Joining (Wire Nuts)

+12V and GND rails are joined with wire nuts (on hand) — no terminal block needed.

```
[+12V wire nut]          [Common GND wire nut]
  IRM-30-12 (+)            IRM-30-12 (−)
  ring generator (+)       ring generator (−)
  carbon mic bias (+)      carbon mic bias (−)
                           Pi GND GPIO pin
```

## Power Budget (Phase 2 Estimate)

| Component | Draw |
|---|---|
| Raspberry Pi 5 under load (includes USB DAC draw through Pi USB port) | ~3,000–5,000mA @ 5V |
| Ring generator module input | ~95mA idle; ~700–900mA @ 12V while ringing (4.6W output at 42–55% efficiency) |
| Carbon mic bias circuit | ~10–50mA @ 12V |
| Relay coil (5V relay) | ~50–100mA @ 5V |
| **Total 5V** | **~3,050–5,100mA** |
| **Total 12V** | **~510–850mA** |

## Antique Hardware Integration Testing

Test incrementally — validate each circuit in isolation before combining. The Mean Well PSUs are not required for bench testing; a 12V wall wart with a shared GND to the Pi is sufficient for the mic bias circuit.

### Step 1: Earpiece (DAC → coil)

**Wiring:** 3.5mm plug Tip → earpiece terminal A; Sleeve → earpiece terminal B. No other components.

Discover the Vantec DAC device index first:

```bash
aplay -l
```

Then drive a 440Hz test tone through it (replace `hw:X,0` with the actual device):

```bash
speaker-test -D hw:X,0 -c 1 -t sine -f 440
```

**Pass:** audible tone in the earpiece. Adjust system volume via `alsamixer` if inaudible.

---

### Step 2: Crank button triggers operator

**Wiring:** momentary button between GPIO pin and GND; enable internal pull-up in software (`PUD_UP`). Confirm the GPIO pin number in the software config before wiring.

Start the AI operator and press the button.

**Pass:** "Number please?" comes through the earpiece.

---

### Step 3: Carbon microphone signal level

Wire the bias + attenuator circuit (see Carbon Microphone Bias section above) and check signal level before running the full stack:

```bash
# Record 5 seconds while speaking into the mic — adjust hw index as above
arecord -D hw:X,0 -f S16_LE -r 44100 -c 1 -d 5 test.wav

# Play back to check for audible speech
aplay test.wav
```

Also inspect levels visually:

```bash
arecord -D hw:X,0 -f S16_LE -r 44100 -c 1 -vv /dev/null
```

The VU meter should show activity when speaking, no clipping.

**Pass:** speech audible in playback, VU meter active, no clipping (bars not pegged at max).

**Tuning the attenuator:** starting values are 10kΩ series / 1kΩ shunt. If clipping, increase the series resistor (try 22kΩ). If signal is too weak, decrease it (try 4.7kΩ). Re-run `arecord -vv` after each change.

**Orientation reminder:** the carbon element must be held in wall-mount orientation during testing — granules shift when inverted and resistance climbs to ~2kΩ, killing signal.

---

### Step 4: Full outbound call path

With all three circuits wired and Steps 1–3 passing:

1. Start the AI operator
2. Press the crank button
3. Speak a contact name or number into the carbon mic
4. Confirm Whisper STT picks up the speech and the operator responds correctly through the earpiece
5. Confirm a SIP call connects and bidirectional audio works

**Pass:** complete flow from button press → operator greeting → STT → LLM response → SIP connect → live audio both directions.

---

## Antique Component Measurements

| Component | Measurement | Notes |
|---|---|---|
| Earpiece | 63Ω DC resistance | Lower than typical POTS-era receivers (200–600Ω); normal for pre-standardization Kellogg design |
| Carbon microphone | ~270Ω (correct orientation) | Orientation-sensitive — granules mobile; must be mounted correctly |
| Magneto crank | ~75–100V AC (estimated) | Multimeter reads ~5V but underreads at 20Hz — actual output confirmed by bell strike. Output disconnected from circuit. |
| Ringer coils | 1604Ω total (2× 800Ω in series) | Center tap terminal unused — connect to outer terminals only |
