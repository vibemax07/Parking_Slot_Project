/* ============================================================
   Smart Parking System — Frontend Application Logic
   Real slot layout matches slots.pkl (41 slots, 4 rows)
   ============================================================ */

'use strict';

// ── Config ──────────────────────────────────────────────────
const API_BASE = 'http://localhost:5000';

// Vehicle definitions
const VEHICLES = {
  hatchback: { label: 'Hatchback', icon: '🚗', desc: 'Compact · Fast entry · Best for tight spots' },
  sedan:     { label: 'Sedan',     icon: '🚙', desc: 'Mid-size · Balanced scoring · Comfort-first' },
  suv:       { label: 'SUV',       icon: '🚕', desc: 'Large · Needs open neighbors · Wide maneuver' },
  truck:     { label: 'Truck',     icon: '🚛', desc: 'Extra-large · Requires adjacent pair of slots' },
};

/*
  Real parking lot layout (derived from slots.pkl):
  Row A: Slots  0–13  (14 slots, horizontal,  top of lot)
  Row B: Slots 14–20  (7 slots,  vertical,    right side)
  Row C: Slots 21–30  (10 slots, horizontal,  middle)
  Row D: Slots 31–40  (10 slots, horizontal,  bottom)
*/
const LAYOUT_ROWS = [
  { label: 'ROW A', ids: [0,1,2,3,4,5,6,7,8,9,10,11,12,13] },
  { label: 'ROW B (RIGHT SIDE)', ids: [14,15,16,17,18,19,20], vertical: true },
  { label: 'ROW C', ids: [30,29,28,27,26,25,24,23,22,21] },   // reversed to match spatial order
  { label: 'ROW D', ids: [31,32,33,34,35,36,37,38,39,40] },
];

// ── State ────────────────────────────────────────────────────
const state = {
  selectedVehicle: null,
  slots:           [],         // [{id, status, x, y, w, h, area}]
  occupiedIds:     new Set(),  // frontend-tracked occupied IDs
  bestSlotId:      null,
  bestPairEndId:   null,
  isAnimating:     false,
  isConnected:     false,
  slotMap:         {},         // id → slot data
};

// ── DOM refs ─────────────────────────────────────────────────
const dom = {
  selectionScreen: document.getElementById('selection-screen'),
  selectionLoader: document.getElementById('selection-loader'),
  dashboard:       document.getElementById('dashboard'),
  vehicleCards:    document.querySelectorAll('.vehicle-card'),

  parkingGrid:     document.getElementById('parking-grid'),
  animatedCar:     document.getElementById('animated-car'),
  roadLane:        document.getElementById('road-lane'),

  statAvailable:   document.getElementById('stat-available'),
  statOccupied:    document.getElementById('stat-occupied'),
  statTotal:       document.getElementById('stat-total'),
  statRate:        document.getElementById('stat-rate'),
  occBarFill:      document.getElementById('occ-bar-fill'),
  occBarPct:       document.getElementById('occ-bar-pct'),

  logBody:         document.getElementById('log-body'),
  logClear:        document.getElementById('log-clear'),

  btnPark:         document.getElementById('btn-park'),
  lotFullBanner:   document.getElementById('lot-full-banner'),
  vehicleBadge:    document.getElementById('vehicle-badge'),
  apiStatus:       document.getElementById('api-status'),
  apiStatusDot:    document.getElementById('api-status-dot'),
  clock:           document.getElementById('clock'),
  toast:           document.getElementById('toast'),
};


// ═══════════════════════════════════════════════════════════
//  CLOCK
// ═══════════════════════════════════════════════════════════
function tickClock() {
  const now = new Date();
  dom.clock.textContent = now.toLocaleTimeString('en-US', { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();


// ═══════════════════════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════
let toastTimer = null;
function showToast(msg, type = 'info', duration = 3500) {
  dom.toast.textContent = msg;
  dom.toast.className   = `show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { dom.toast.className = ''; }, duration);
}


// ═══════════════════════════════════════════════════════════
//  ACTIVITY LOG
// ═══════════════════════════════════════════════════════════
function addLog(msg, type = 'info') {
  const time  = new Date().toLocaleTimeString('en-US', { hour12: false });
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.innerHTML = `<span class="log-time">[${time}]</span><span>${msg}</span>`;
  dom.logBody.appendChild(entry);
  dom.logBody.scrollTop = dom.logBody.scrollHeight;
  while (dom.logBody.children.length > 100) {
    dom.logBody.removeChild(dom.logBody.firstChild);
  }
}

dom.logClear.addEventListener('click', () => {
  dom.logBody.innerHTML = '';
  addLog('Log cleared.', 'system');
});


// ═══════════════════════════════════════════════════════════
//  STATS
// ═══════════════════════════════════════════════════════════
function updateStats(available, occupied, total) {
  const pct = total > 0 ? Math.round((occupied / total) * 100) : 0;
  dom.statAvailable.textContent = available;
  dom.statOccupied.textContent  = occupied;
  dom.statTotal.textContent     = total;
  dom.statRate.textContent      = `${pct}%`;
  dom.occBarFill.style.width    = `${pct}%`;
  dom.occBarPct.textContent     = `${pct}%`;
}


// ═══════════════════════════════════════════════════════════
//  API HEALTH CHECK
// ═══════════════════════════════════════════════════════════
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      state.isConnected = true;
      dom.apiStatus.textContent          = 'API ONLINE';
      dom.apiStatusDot.style.background  = 'var(--accent-green)';
      return true;
    }
  } catch (_) { /* fall through */ }
  state.isConnected = false;
  dom.apiStatus.textContent         = 'API OFFLINE';
  dom.apiStatusDot.style.background = 'var(--accent-red)';
  return false;
}
setInterval(checkApiHealth, 15000);


// ═══════════════════════════════════════════════════════════
//  PARKING GRID RENDERING  (41 slots, 4 rows)
// ═══════════════════════════════════════════════════════════
function buildGrid(slots) {
  // Build a lookup map
  state.slotMap = {};
  slots.forEach(s => { state.slotMap[s.id] = s; });

  dom.parkingGrid.innerHTML = '';

  LAYOUT_ROWS.forEach(rowDef => {
    const rowWrap  = document.createElement('div');
    rowWrap.className = 'parking-row-wrap';

    const rowLabel = document.createElement('div');
    rowLabel.className = 'row-label';
    rowLabel.textContent = rowDef.label;
    rowWrap.appendChild(rowLabel);

    const rowGrid = document.createElement('div');

    if (rowDef.vertical) {
      // Row B: stack vertically, 2 columns of 4+3
      rowGrid.className   = 'slots-row vertical-row';
      rowGrid.style.gridTemplateColumns = 'repeat(4, 1fr)';
    } else {
      rowGrid.className   = 'slots-row';
      rowGrid.style.gridTemplateColumns = `repeat(${rowDef.ids.length}, 1fr)`;
    }

    rowDef.ids.forEach(id => {
      const slot = state.slotMap[id];
      if (!slot) return;
      rowGrid.appendChild(createSlotElement(slot));
    });

    rowWrap.appendChild(rowGrid);
    dom.parkingGrid.appendChild(rowWrap);
  });
}

function slotStatusClass(slot) {
  const isOccupied = slot.status === 'occupied' || state.occupiedIds.has(slot.id);
  const isBest     = slot.id === state.bestSlotId || slot.id === state.bestPairEndId;
  if (isBest && !state.occupiedIds.has(slot.id))  return 'best';
  if (isOccupied)                                  return 'occupied';
  return 'empty';
}

function createSlotElement(slot) {
  const cls  = slotStatusClass(slot);
  const icon = cls === 'best' ? '⭐' : cls === 'occupied' ? '🚗' : '✓';
  const text = cls === 'best' ? 'BEST' : cls === 'occupied' ? 'OCCUPIED' : 'FREE';

  const el = document.createElement('div');
  el.className      = `slot ${cls}`;
  el.id             = `slot-${slot.id}`;
  el.dataset.slotId = slot.id;
  el.title          = `P${slot.id + 1} · ${text}`;

  el.innerHTML = `
    <div class="slot-id">P${slot.id + 1}</div>
    <div class="slot-status-icon">${icon}</div>
    <div class="slot-status-text">${text}</div>
  `;
  return el;
}

function markSlotOccupied(slotId) {
  const el = document.getElementById(`slot-${slotId}`);
  if (!el) return;
  el.className = 'slot occupied just-parked';
  el.innerHTML = `
    <div class="slot-id">P${slotId + 1}</div>
    <div class="slot-status-icon">🚗</div>
    <div class="slot-status-text">OCCUPIED</div>
  `;
  setTimeout(() => el.classList.remove('just-parked'), 900);
}

function highlightBestSlot(slotId) {
  // Clear any previous best highlight
  document.querySelectorAll('.slot.best').forEach(el => {
    const id = parseInt(el.dataset.slotId);
    const s  = state.slotMap[id];
    if (s) {
      el.className = `slot ${state.occupiedIds.has(id) ? 'occupied' : 'empty'}`;
      const ic = state.occupiedIds.has(id) ? '🚗' : '✓';
      const tx = state.occupiedIds.has(id) ? 'OCCUPIED' : 'FREE';
      el.innerHTML = `<div class="slot-id">P${id+1}</div><div class="slot-status-icon">${ic}</div><div class="slot-status-text">${tx}</div>`;
    }
  });

  const el = document.getElementById(`slot-${slotId}`);
  if (!el) return;
  el.className = 'slot best';
  el.innerHTML = `
    <div class="slot-id">P${slotId + 1}</div>
    <div class="slot-status-icon">⭐</div>
    <div class="slot-status-text">BEST</div>
  `;
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


// ═══════════════════════════════════════════════════════════
//  CAR ANIMATION
// ═══════════════════════════════════════════════════════════
function animateCar(vehicleIcon, targetSlotId, onComplete) {
  const car      = dom.animatedCar;
  const road     = dom.roadLane;
  const slotEl   = document.getElementById(`slot-${targetSlotId}`);

  car.textContent       = vehicleIcon;
  car.style.transition  = 'none';
  car.style.left        = '-80px';
  car.style.top         = `${road.offsetHeight / 2 - 18}px`;
  car.style.opacity     = '1';
  car.style.transform   = '';

  // Phase 1: drive along road
  requestAnimationFrame(() => {
    car.style.transition = 'left 1.6s cubic-bezier(0.4,0,0.2,1)';
    // Drive to ~65% of road width
    car.style.left       = `${road.offsetWidth * 0.65}px`;

    setTimeout(() => {
      // Phase 2: move towards target slot (if visible)
      if (slotEl) {
        const roadRect = road.getBoundingClientRect();
        const slotRect = slotEl.getBoundingClientRect();
        const targetLeft = slotRect.left - roadRect.left + slotRect.width / 2 - 18;

        car.style.transition = 'left 0.7s ease-in-out';
        car.style.left       = `${Math.max(0, Math.min(road.offsetWidth - 40, targetLeft))}px`;
      }

      setTimeout(() => {
        // Phase 3: park (rotate + shrink)
        car.style.transition = 'transform 0.5s ease, opacity 0.45s ease';
        car.style.transform  = 'rotate(90deg) scale(0.8)';

        setTimeout(() => {
          car.style.opacity   = '0';
          car.style.transform = 'rotate(90deg) scale(0)';
          setTimeout(() => {
            car.style.transition = 'none';
            car.style.transform  = '';
            car.style.left       = '-80px';
            car.style.opacity    = '0';
            onComplete();
          }, 420);
        }, 520);
      }, 800);
    }, 1700);
  });
}


// ═══════════════════════════════════════════════════════════
//  DEMO MODE (no backend)
// ═══════════════════════════════════════════════════════════
function buildDemoSlots() {
  const total       = 41;
  const demoOccupied = new Set([2, 5, 8, 14, 15, 22, 25, 31, 36]);
  const minArea     = { hatchback: 1500, sedan: 2500, suv: 3500, truck: 3500 }[state.selectedVehicle] || 1500;

  state.slots = Array.from({ length: total }, (_, i) => ({
    id:     i,
    status: (demoOccupied.has(i) || state.occupiedIds.has(i)) ? 'occupied' : 'empty',
    x: 0, y: 0, w: 110, h: 230, area: 25300,
  }));

  const candidate    = state.slots.find(s => s.status === 'empty' && s.area >= minArea);
  state.bestSlotId   = candidate ? candidate.id : null;
  state.bestPairEndId = null;

  const available = state.slots.filter(s => s.status === 'empty').length;
  buildGrid(state.slots);
  updateStats(available, total - available, total);
  addLog(`Demo mode: ${total} simulated slots loaded.`, 'system');
}


// ═══════════════════════════════════════════════════════════
//  VEHICLE SELECTION → DASHBOARD TRANSITION
// ═══════════════════════════════════════════════════════════
dom.vehicleCards.forEach(card => {
  card.addEventListener('click', () => {
    if (dom.selectionLoader.classList.contains('active')) return;
    selectVehicle(card.dataset.vehicle);
  });
  // Keyboard support
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      card.click();
    }
  });
});

async function selectVehicle(vehicleType) {
  state.selectedVehicle = vehicleType;
  const veh = VEHICLES[vehicleType];
  dom.selectionLoader.classList.add('active');
  addLog(`Vehicle selected: ${veh.label.toUpperCase()}`, 'info');

  const online = await checkApiHealth();

  if (!online) {
    addLog('⚠ Backend offline — using demo mode.', 'warn');
    showToast('Backend offline — start app.py · Demo mode active', 'warn', 5000);
    buildDemoSlots();
    transitionToDashboard(vehicleType);
    dom.selectionLoader.classList.remove('active');
    return;
  }

  addLog('Scanning parking lot via ML model…', 'info');

  try {
    const res  = await fetch(`${API_BASE}/best-slot`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        car_type:     vehicleType,
        occupied_ids: [...state.occupiedIds],
      }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    state.slots         = data.slots;
    state.bestSlotId    = data.best_slot_id;
    state.bestPairEndId = data.best_pair_end;

    buildGrid(state.slots);
    updateStats(data.available, data.occupied, data.total);
    addLog(`Scan complete — ${data.available}/${data.total} slots available`, 'success');

    if (data.best_slot_id !== null) {
      addLog(`Optimal slot for ${veh.label}: P${data.best_slot_id + 1}`, 'success');
    } else {
      addLog(`⚠ No suitable slot for ${veh.label}`, 'warn');
    }

  } catch (err) {
    addLog(`API error: ${err.message} — falling back to demo.`, 'error');
    buildDemoSlots();
  }

  transitionToDashboard(vehicleType);
  dom.selectionLoader.classList.remove('active');
}

function transitionToDashboard(vehicleType) {
  const veh = VEHICLES[vehicleType];
  dom.vehicleBadge.textContent = `${veh.icon}  ${veh.label.toUpperCase()}`;
  dom.selectionScreen.classList.add('hidden');
  dom.dashboard.classList.add('visible');
  dom.animatedCar.style.opacity = '0';

  dom.lotFullBanner.classList.remove('visible');

  if (state.bestSlotId !== null) {
    highlightBestSlot(state.bestSlotId);
    dom.btnPark.disabled    = false;
    dom.btnPark.textContent = `▶  PARK ${veh.label.toUpperCase()} · SLOT P${state.bestSlotId + 1}`;
    showToast(`Slot P${state.bestSlotId + 1} recommended for your ${veh.label}`, 'success');
  } else {
    dom.btnPark.disabled    = true;
    dom.btnPark.textContent = '⊘  NO SUITABLE SLOT';
    dom.lotFullBanner.classList.add('visible');
    showToast('No suitable slot available for this vehicle', 'error', 5000);
  }
}


// ═══════════════════════════════════════════════════════════
//  PARK BUTTON — single unified handler
// ═══════════════════════════════════════════════════════════
dom.btnPark.addEventListener('click', async () => {
  if (state.isAnimating || dom.btnPark.disabled) return;
  const veh = VEHICLES[state.selectedVehicle];

  // If we already have a best slot designated, park there
  if (state.bestSlotId !== null) {
    const slotId = state.bestSlotId;
    state.isAnimating   = true;
    dom.btnPark.disabled = true;
    dom.btnPark.textContent = '⏳  PARKING…';

    addLog(`Initiating parking → P${slotId + 1}`, 'info');
    showToast(`Parking ${veh.label} in slot P${slotId + 1}…`, 'info');

    animateCar(veh.icon, slotId, async () => {
      // Mark occupied
      state.occupiedIds.add(slotId);
      if (state.slotMap[slotId]) state.slotMap[slotId].status = 'occupied';
      state.slots.forEach(s => { if (s.id === slotId) s.status = 'occupied'; });
      markSlotOccupied(slotId);

      // Also mark pair-end if truck
      if (state.bestPairEndId !== null) {
        state.occupiedIds.add(state.bestPairEndId);
        markSlotOccupied(state.bestPairEndId);
      }

      const available = state.slots.filter(s => s.status === 'empty' && !state.occupiedIds.has(s.id)).length;
      const occupied  = state.slots.length - available;
      updateStats(available, occupied, state.slots.length);

      addLog(`✓ ${veh.label.toUpperCase()} parked in P${slotId + 1}`, 'success');
      addLog(`Stats → Available: ${available}/${state.slots.length}`, 'system');
      showToast(`${veh.label} parked in P${slotId + 1}!`, 'success');

      state.isAnimating   = false;
      state.bestSlotId    = null;
      state.bestPairEndId = null;

      if (available === 0) {
        dom.lotFullBanner.classList.add('visible');
        dom.btnPark.textContent = '⊘  LOT FULL';
        dom.btnPark.disabled    = true;
        addLog('⚠ Parking lot FULL', 'warn');
        showToast('Parking lot is full!', 'error', 5000);
        return;
      }

      // Auto-find next best slot
      addLog('Finding next best slot…', 'info');
      await findNextBestSlot();
    });

  } else {
    // Should not happen but just in case
    await findNextBestSlot();
  }
});

async function findNextBestSlot() {
  const veh = VEHICLES[state.selectedVehicle];
  try {
    let bestId = null;
    let pairEnd = null;

    if (state.isConnected) {
      const res  = await fetch(`${API_BASE}/best-slot`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          car_type:     state.selectedVehicle,
          occupied_ids: [...state.occupiedIds],
        }),
      });
      const data = await res.json();
      if (!data.error) {
        state.slots = data.slots;
        // Re-apply frontend tracked occupied
        state.slots.forEach(s => { if (state.occupiedIds.has(s.id)) s.status = 'occupied'; });
        buildGrid(state.slots);
        updateStats(
          state.slots.filter(s => s.status === 'empty').length,
          state.occupiedIds.size,
          state.slots.length
        );
        bestId  = data.best_slot_id;
        pairEnd = data.best_pair_end;
      }
    } else {
      // Demo mode: find first suitable empty slot
      const minArea = { hatchback: 1500, sedan: 2500, suv: 3500, truck: 3500 }[state.selectedVehicle] || 1500;
      const candidate = state.slots.find(s => s.status === 'empty' && !state.occupiedIds.has(s.id) && s.area >= minArea);
      bestId = candidate ? candidate.id : null;
      buildGrid(state.slots);
    }

    state.bestSlotId    = bestId;
    state.bestPairEndId = pairEnd;

    if (bestId !== null) {
      highlightBestSlot(bestId);
      dom.btnPark.disabled    = false;
      dom.btnPark.textContent = `▶  PARK ${veh.label.toUpperCase()} · SLOT P${bestId + 1}`;
      addLog(`Next best slot: P${bestId + 1}`, 'success');
      showToast(`Next slot ready: P${bestId + 1}`, 'success');
    } else {
      dom.btnPark.disabled    = true;
      dom.btnPark.textContent = '⊘  NO SUITABLE SLOT';
      dom.lotFullBanner.classList.add('visible');
      addLog('No more suitable slots available.', 'warn');
    }
  } catch (err) {
    addLog(`Error finding next slot: ${err.message}`, 'error');
    dom.btnPark.disabled    = false;
    dom.btnPark.textContent = `▶  PARK ${veh.label.toUpperCase()}`;
  }
}


// ═══════════════════════════════════════════════════════════
//  RESET & BACK
// ═══════════════════════════════════════════════════════════
document.getElementById('btn-reset').addEventListener('click', () => {
  state.occupiedIds.clear();
  state.bestSlotId      = null;
  state.bestPairEndId   = null;
  state.isAnimating     = false;
  state.selectedVehicle = null;
  state.slots           = [];
  state.slotMap         = {};

  dom.selectionLoader.classList.remove('active');
  dom.selectionScreen.classList.remove('hidden');
  dom.dashboard.classList.remove('visible');
  dom.lotFullBanner.classList.remove('visible');
  dom.btnPark.disabled    = true;
  dom.btnPark.textContent = '▶  SELECT VEHICLE FIRST';
  dom.vehicleBadge.textContent = '— NO VEHICLE —';

  dom.logBody.innerHTML = '';
  updateStats(0, 0, 0);
  dom.parkingGrid.innerHTML = '';
  dom.animatedCar.style.opacity = '0';
  addLog('System reset by user.', 'system');
});

document.getElementById('btn-back').addEventListener('click', () => {
  document.getElementById('btn-reset').click();
});


// ═══════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════
(async function boot() {
  addLog('Smart Parking System v1.0 — Initialising…', 'system');
  addLog('Checking ML backend…', 'info');
  const online = await checkApiHealth();
  if (online) {
    addLog('Backend API connected ✓ (41 slots ready)', 'success');
    showToast('Backend connected — AI model loaded', 'success', 3000);
  } else {
    addLog('Backend offline — demo mode available when a vehicle is selected.', 'warn');
    showToast('Start app.py to enable AI detection · Demo mode available', 'warn', 6000);
  }
  addLog('Select a vehicle type above to begin.', 'info');
})();
