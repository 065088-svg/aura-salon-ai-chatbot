"""
booking_engine.py
-----------------
Deterministic booking logic for "Aura Salon & Spa".

DESIGN RULE: the LLM never decides whether a slot is free or a booking exists.
Every fact (availability, booking IDs, policy outcomes) comes from this module;
the LLM only handles conversation and calls these functions as tools.
"""
from __future__ import annotations

import difflib
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Kolkata")

# ----------------------------------------------------------------- business config
BUSINESS = {
    "name": "Aura Salon & Spa",
    "address": "Sector 15, Faridabad (demo address)",
    "front_desk_phone": "+91-00000-00000 (demo)",
    "hours": "Mon-Sat, 10:00-19:00 (closed Sundays)",
}
SERVICES = {
    "Haircut": {"minutes": 30, "price": 500},
    "Beard Trim": {"minutes": 30, "price": 300},
    "Manicure": {"minutes": 60, "price": 700},
    "Facial": {"minutes": 60, "price": 1500},
    "Hair Colour": {"minutes": 120, "price": 2500},
}
STAFF = {
    "Riya": ["Haircut", "Hair Colour", "Facial", "Manicure"],
    "Aman": ["Haircut", "Beard Trim"],
    "Neha": ["Facial", "Manicure", "Hair Colour"],
}
ALIASES = {
    "hair cut": "Haircut", "cut": "Haircut", "beard": "Beard Trim", "shave": "Beard Trim",
    "color": "Hair Colour", "colour": "Hair Colour", "hair color": "Hair Colour",
    "coloring": "Hair Colour", "colouring": "Hair Colour", "mani": "Manicure", "nails": "Manicure",
    "face": "Facial", "facial": "Facial",
}
OPEN_H, CLOSE_H = 10, 19
SLOT_MIN = 30
LEAD_MIN = 60              # need >= 60 min notice for a new booking
MAX_DAYS_AHEAD = 30
FREE_CANCEL_HOURS = 4      # cancel/reschedule earlier than this = free
GRACE_MIN = 15             # no-show only after 15 min grace
NOSHOW_LIMIT = 2           # 2+ no-shows -> advance deposit
DEPOSIT_INR = 200

POLICIES = {
    "hours": BUSINESS["hours"],
    "booking_notice": f"Bookings need at least {LEAD_MIN} minutes' notice and can be made up to {MAX_DAYS_AHEAD} days ahead.",
    "cancellation": f"Free cancel/reschedule up to {FREE_CANCEL_HOURS} hours before the appointment. Later changes are logged as 'late cancellation'.",
    "grace_period": f"Please arrive on time. The slot is released and marked a no-show if you are more than {GRACE_MIN} minutes late with no notice.",
    "no_show": f"After {NOSHOW_LIMIT} no-shows, a refundable Rs {DEPOSIT_INR} advance is required for future bookings.",
    "reminders": "A confirmation is sent on booking and a reminder about 24 hours before the appointment.",
    "payment": "Pay at the salon after the service (UPI/cash/card).",
}


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%a %d %b %Y, %I:%M %p").replace(" 0", " ")


class Engine:
    def __init__(self, path: str = ":memory:", clock=None):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.clock = clock or (lambda: datetime.now(TZ).replace(tzinfo=None))
        self._init_db()

    # ------------------------------------------------------------------ schema
    def _init_db(self):
        with self.lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS bookings(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT, phone TEXT, service TEXT, stylist TEXT,
                    start TEXT, minutes INTEGER,
                    status TEXT DEFAULT 'booked',      -- booked | cancelled | no_show
                    late_cancel INTEGER DEFAULT 0,
                    reminded INTEGER DEFAULT 0,
                    created_at TEXT);
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    booking_id TEXT, phone TEXT, kind TEXT, text TEXT, created_at TEXT);
                CREATE TABLE IF NOT EXISTS handoffs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT, phone TEXT, reason TEXT, created_at TEXT);
                """
            )
            self.db.commit()

    def reset(self):
        with self.lock:
            self.db.executescript("DROP TABLE IF EXISTS bookings; DROP TABLE IF EXISTS messages; DROP TABLE IF EXISTS handoffs;")
            self._init_db()
            self.seed()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def bid(row_id: int) -> str:
        return f"AUR-{1000 + row_id}"

    @staticmethod
    def parse_bid(s: str):
        m = re.fullmatch(r"AUR-?(\d{4,})", (s or "").strip().upper().replace(" ", ""))
        return int(m.group(1)) - 1000 if m else None

    @staticmethod
    def clean_phone(p: str):
        digits = re.sub(r"\D", "", p or "")
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None

    @staticmethod
    def norm_service(name: str):
        s = " ".join((name or "").lower().split())
        for k in SERVICES:
            if s == k.lower():
                return k
        if s in ALIASES:
            return ALIASES[s]
        close = difflib.get_close_matches(s, [k.lower() for k in SERVICES], n=1, cutoff=0.75)
        if close:
            return next(k for k in SERVICES if k.lower() == close[0])
        return None

    @staticmethod
    def _parse_dt(date_s: str, time_s: str):
        try:
            return datetime.strptime(f"{date_s.strip()} {time_s.strip()}", "%Y-%m-%d %H:%M")
        except (ValueError, AttributeError):
            return None

    def _d(self, row) -> dict:
        start = datetime.fromisoformat(row["start"])
        return {
            "booking_id": self.bid(row["id"]), "name": row["name"], "phone": row["phone"],
            "service": row["service"], "stylist": row["stylist"], "start": row["start"],
            "when": fmt_dt(start), "minutes": row["minutes"], "status": row["status"],
            "price_inr": SERVICES[row["service"]]["price"],
        }

    def _slot_error(self, start: datetime, minutes: int):
        now = self.clock()
        if start.weekday() == 6:
            return "The salon is closed on Sundays."
        if start.minute % SLOT_MIN:
            return "Appointments start on the hour or half hour (e.g. 15:00 or 15:30)."
        if start.hour < OPEN_H or start + timedelta(minutes=minutes) > start.replace(hour=CLOSE_H, minute=0):
            return f"Outside working hours ({OPEN_H}:00-{CLOSE_H}:00). This {minutes}-minute service must finish by {CLOSE_H}:00."
        if start < now + timedelta(minutes=LEAD_MIN):
            return f"That time is in the past or less than {LEAD_MIN} minutes away. Please pick a later slot."
        if start.date() > now.date() + timedelta(days=MAX_DAYS_AHEAD):
            return f"We only take bookings up to {MAX_DAYS_AHEAD} days ahead."
        return None

    def _is_free(self, stylist: str, start: datetime, minutes: int, exclude_row=None) -> bool:
        end = start + timedelta(minutes=minutes)
        rows = self.db.execute(
            "SELECT id,start,minutes FROM bookings WHERE stylist=? AND status='booked' AND substr(start,1,10)=?",
            (stylist, start.strftime("%Y-%m-%d")),
        ).fetchall()
        for r in rows:
            if exclude_row is not None and r["id"] == exclude_row:
                continue
            s = datetime.fromisoformat(r["start"])
            if s < end and s + timedelta(minutes=r["minutes"]) > start:
                return False
        return True

    def _free_stylists(self, service, start, minutes, preferred=None, exclude_row=None):
        names = [n for n, svcs in STAFF.items() if service in svcs]
        if preferred:
            names = [n for n in names if n == preferred]
        return [n for n in names if self._is_free(n, start, minutes, exclude_row)]

    def _day_load(self, stylist, day: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) c FROM bookings WHERE stylist=? AND status='booked' AND substr(start,1,10)=?", (stylist, day)
        ).fetchone()["c"]

    def _log_msg(self, booking_id, phone, kind, text):
        self.db.execute(
            "INSERT INTO messages(booking_id,phone,kind,text,created_at) VALUES(?,?,?,?,?)",
            (booking_id, phone, kind, text, self.clock().isoformat(timespec="minutes")),
        )

    def noshow_count(self, phone: str) -> int:
        return self.db.execute("SELECT COUNT(*) c FROM bookings WHERE phone=? AND status='no_show'", (phone,)).fetchone()["c"]

    @staticmethod
    def _norm_stylist(s):
        s = (s or "").strip().title()
        return s if s in STAFF else None

    # ------------------------------------------------------------- public API
    def list_services(self) -> dict:
        return {
            "ok": True,
            "services": [
                {"service": k, "minutes": v["minutes"], "price_inr": v["price"],
                 "stylists": [n for n, s in STAFF.items() if k in s]}
                for k, v in SERVICES.items()
            ],
        }

    def get_policies(self) -> dict:
        return {"ok": True, "business": BUSINESS, "policies": POLICIES}

    def check_availability(self, service: str, date: str, stylist: str = "") -> dict:
        svc = self.norm_service(service)
        if not svc:
            return {"ok": False, "error": f"Unknown service '{service}'.", "available_services": list(SERVICES)}
        try:
            day = datetime.strptime(date.strip(), "%Y-%m-%d")
        except (ValueError, AttributeError):
            return {"ok": False, "error": "Date must be in YYYY-MM-DD format."}
        pref = None
        if stylist:
            pref = self._norm_stylist(stylist)
            if not pref:
                return {"ok": False, "error": f"No stylist named '{stylist}'.", "stylists": list(STAFF)}
            if svc not in STAFF[pref]:
                return {"ok": False, "error": f"{pref} does not offer {svc}.",
                        "stylists_for_service": [n for n, s in STAFF.items() if svc in s]}
        minutes = SERVICES[svc]["minutes"]
        with self.lock:
            slots = []
            t = day.replace(hour=OPEN_H, minute=0)
            while t.hour < CLOSE_H:
                if self._slot_error(t, minutes) is None:
                    free = self._free_stylists(svc, t, minutes, pref)
                    if free:
                        slots.append({"time": t.strftime("%H:%M"), "stylists": free})
                t += timedelta(minutes=SLOT_MIN)
            res = {"ok": True, "service": svc, "date": date, "weekday": day.strftime("%A"),
                   "duration_minutes": minutes, "available_slots": slots}
            if not slots:
                res["note"] = "No availability on this date."
                res["next_dates_with_availability"] = self._next_dates(svc, day, pref)
            return res

    def _next_dates(self, svc, day, pref, n=3):
        out, d, minutes = [], day, SERVICES[svc]["minutes"]
        for _ in range(MAX_DAYS_AHEAD):
            d += timedelta(days=1)
            t = d.replace(hour=OPEN_H, minute=0)
            while t.hour < CLOSE_H:
                if self._slot_error(t, minutes) is None and self._free_stylists(svc, t, minutes, pref):
                    out.append(d.strftime("%Y-%m-%d (%a)"))
                    break
                t += timedelta(minutes=SLOT_MIN)
            if len(out) >= n:
                break
        return out

    def book(self, name: str, phone: str, service: str, date: str, time: str, stylist: str = "") -> dict:
        name = " ".join((name or "").split())
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,59}", name):
            return {"ok": False, "error": "Please provide the customer's name (letters only, 2-60 characters)."}
        ph = self.clean_phone(phone)
        if not ph:
            return {"ok": False, "error": "Please provide a valid 10-digit Indian mobile number."}
        svc = self.norm_service(service)
        if not svc:
            return {"ok": False, "error": f"Unknown service '{service}'.", "available_services": list(SERVICES)}
        start = self._parse_dt(date, time)
        if not start:
            return {"ok": False, "error": "Date must be YYYY-MM-DD and time HH:MM (24-hour)."}
        minutes = SERVICES[svc]["minutes"]
        err = self._slot_error(start, minutes)
        if err:
            return {"ok": False, "error": err}
        pref = None
        if stylist:
            pref = self._norm_stylist(stylist)
            if not pref:
                return {"ok": False, "error": f"No stylist named '{stylist}'.", "stylists": list(STAFF)}
            if svc not in STAFF[pref]:
                return {"ok": False, "error": f"{pref} does not offer {svc}."}
        with self.lock:
            # duplicate / double-submit guard: same phone cannot hold overlapping appointments
            end = start + timedelta(minutes=minutes)
            for r in self.db.execute("SELECT * FROM bookings WHERE phone=? AND status='booked'", (ph,)).fetchall():
                s = datetime.fromisoformat(r["start"])
                if s < end and s + timedelta(minutes=r["minutes"]) > start:
                    return {"ok": False, "error": "This phone number already has an overlapping booking.",
                            "existing_booking": self._d(r)}
            free = self._free_stylists(svc, start, minutes, pref)
            if not free:
                alt = self.check_availability(svc, start.strftime("%Y-%m-%d"), pref or "")
                return {"ok": False, "error": "That slot is no longer available" + (f" with {pref}." if pref else "."),
                        "other_slots_that_day": alt.get("available_slots", [])[:6]}
            chosen = min(free, key=lambda n: self._day_load(n, start.strftime("%Y-%m-%d")))
            cur = self.db.execute(
                "INSERT INTO bookings(name,phone,service,stylist,start,minutes,created_at) VALUES(?,?,?,?,?,?,?)",
                (name, ph, svc, chosen, start.strftime("%Y-%m-%dT%H:%M"), minutes, self.clock().isoformat(timespec="minutes")),
            )
            bid = self.bid(cur.lastrowid)
            text = (f"Hi {name.split()[0]}, your {svc} at {BUSINESS['name']} is confirmed for {fmt_dt(start)} "
                    f"with {chosen}. Booking ID: {bid}. Free cancel/reschedule up to {FREE_CANCEL_HOURS}h before. "
                    f"Please arrive 5 min early.")
            self._log_msg(bid, ph, "confirmation", text)
            self.db.commit()
            row = self.db.execute("SELECT * FROM bookings WHERE id=?", (cur.lastrowid,)).fetchone()
            res = {"ok": True, "booking": self._d(row), "confirmation_message_sent": True,
                   "policy_reminder": POLICIES["cancellation"]}
            if self.noshow_count(ph) >= NOSHOW_LIMIT:
                res["deposit_required"] = True
                res["deposit_note"] = (f"This number has {self.noshow_count(ph)} earlier no-shows, so a refundable "
                                       f"Rs {DEPOSIT_INR} advance is required to keep the slot (payment link sent by front desk).")
            return res

    def _get_owned(self, booking_id: str, phone: str):
        rid, ph = self.parse_bid(booking_id), self.clean_phone(phone)
        if rid is None or not ph:
            return None
        row = self.db.execute("SELECT * FROM bookings WHERE id=? AND phone=?", (rid, ph)).fetchone()
        return row

    NOT_FOUND = {"ok": False, "error": "No booking matches that ID and phone number. Please check both."}

    def find_bookings(self, phone: str, booking_id: str = "") -> dict:
        ph = self.clean_phone(phone)
        if not ph:
            return {"ok": False, "error": "Please provide a valid 10-digit mobile number."}
        with self.lock:
            if booking_id:
                row = self._get_owned(booking_id, ph)
                return {"ok": True, "bookings": [self._d(row)]} if row else dict(self.NOT_FOUND)
            rows = self.db.execute(
                "SELECT * FROM bookings WHERE phone=? AND status='booked' AND start>=? ORDER BY start",
                (ph, self.clock().strftime("%Y-%m-%dT%H:%M")),
            ).fetchall()
            return {"ok": True, "bookings": [self._d(r) for r in rows],
                    "note": "No upcoming bookings for this number." if not rows else ""}

    def cancel(self, booking_id: str, phone: str) -> dict:
        with self.lock:
            row = self._get_owned(booking_id, phone)
            if not row:
                return dict(self.NOT_FOUND)
            if row["status"] != "booked":
                return {"ok": False, "error": f"This booking is already {row['status']}."}
            start = datetime.fromisoformat(row["start"])
            if start <= self.clock():
                return {"ok": False, "error": "This appointment time has already passed."}
            late = (start - self.clock()) < timedelta(hours=FREE_CANCEL_HOURS)
            self.db.execute("UPDATE bookings SET status='cancelled', late_cancel=? WHERE id=?", (int(late), row["id"]))
            bid = self.bid(row["id"])
            self._log_msg(bid, row["phone"], "cancellation",
                          f"Hi {row['name'].split()[0]}, your {row['service']} on {fmt_dt(start)} ({bid}) is cancelled. "
                          f"{'Note: this was a late cancellation (under ' + str(FREE_CANCEL_HOURS) + 'h notice). ' if late else ''}"
                          "Hope to see you again soon!")
            self.db.commit()
            return {"ok": True, "booking_id": bid, "status": "cancelled", "late_cancellation": late,
                    "note": f"Late cancellation logged (under {FREE_CANCEL_HOURS}h notice)." if late else "Cancelled free of charge."}

    def reschedule(self, booking_id: str, phone: str, new_date: str, new_time: str) -> dict:
        with self.lock:
            row = self._get_owned(booking_id, phone)
            if not row:
                return dict(self.NOT_FOUND)
            if row["status"] != "booked":
                return {"ok": False, "error": f"This booking is already {row['status']}; it cannot be rescheduled."}
            old_start = datetime.fromisoformat(row["start"])
            if old_start <= self.clock():
                return {"ok": False, "error": "This appointment time has already passed."}
            start = self._parse_dt(new_date, new_time)
            if not start:
                return {"ok": False, "error": "Date must be YYYY-MM-DD and time HH:MM (24-hour)."}
            minutes = row["minutes"]
            err = self._slot_error(start, minutes)
            if err:
                return {"ok": False, "error": err}
            free = self._free_stylists(row["service"], start, minutes, None, exclude_row=row["id"])
            if not free:
                alt = self.check_availability(row["service"], start.strftime("%Y-%m-%d"))
                return {"ok": False, "error": "That new slot is not available. Original booking is unchanged.",
                        "other_slots_that_day": alt.get("available_slots", [])[:6]}
            chosen = row["stylist"] if row["stylist"] in free else free[0]
            late = (old_start - self.clock()) < timedelta(hours=FREE_CANCEL_HOURS)
            self.db.execute("UPDATE bookings SET start=?, stylist=?, reminded=0, late_cancel=? WHERE id=?",
                            (start.strftime("%Y-%m-%dT%H:%M"), chosen, int(late), row["id"]))
            bid = self.bid(row["id"])
            self._log_msg(bid, row["phone"], "reschedule",
                          f"Hi {row['name'].split()[0]}, your {row['service']} ({bid}) is moved to {fmt_dt(start)} with {chosen}.")
            self.db.commit()
            new_row = self.db.execute("SELECT * FROM bookings WHERE id=?", (row["id"],)).fetchone()
            return {"ok": True, "booking": self._d(new_row), "stylist_changed": chosen != row["stylist"],
                    "late_change": late,
                    "note": f"Changed with under {FREE_CANCEL_HOURS}h notice (logged)." if late else ""}

    def create_handoff(self, name: str, phone: str, reason: str) -> dict:
        with self.lock:
            cur = self.db.execute("INSERT INTO handoffs(name,phone,reason,created_at) VALUES(?,?,?,?)",
                                  ((name or "")[:60], self.clean_phone(phone) or (phone or "")[:15], (reason or "")[:300],
                                   self.clock().isoformat(timespec="minutes")))
            self.db.commit()
            return {"ok": True, "ticket_id": f"HO-{cur.lastrowid:03d}",
                    "message": f"A front-desk team member will call back. You can also call {BUSINESS['front_desk_phone']} ({BUSINESS['hours']})."}

    # ------------------------------------------------- front-desk (not LLM tools)
    def run_reminders(self, within_hours: int = 24) -> int:
        now = self.clock()
        with self.lock:
            rows = self.db.execute("SELECT * FROM bookings WHERE status='booked' AND reminded=0 AND start>=? AND start<=? ORDER BY start",
                                   (now.strftime("%Y-%m-%dT%H:%M"), (now + timedelta(hours=within_hours)).strftime("%Y-%m-%dT%H:%M"))).fetchall()
            for r in rows:
                bid, start = self.bid(r["id"]), datetime.fromisoformat(r["start"])
                self._log_msg(bid, r["phone"], "reminder",
                              f"Reminder: {r['name'].split()[0]}, your {r['service']} is on {fmt_dt(start)} with {r['stylist']} ({bid}). "
                              f"Need to change it? Free until {FREE_CANCEL_HOURS}h before. Please arrive on time - we hold the slot {GRACE_MIN} min.")
                self.db.execute("UPDATE bookings SET reminded=1 WHERE id=?", (r["id"],))
            self.db.commit()
            return len(rows)

    def mark_no_show(self, booking_id: str, force: bool = False) -> dict:
        with self.lock:
            rid = self.parse_bid(booking_id)
            row = self.db.execute("SELECT * FROM bookings WHERE id=?", (rid,)).fetchone() if rid is not None else None
            if not row or row["status"] != "booked":
                return {"ok": False, "error": "Booking not found or not in 'booked' state."}
            start = datetime.fromisoformat(row["start"])
            if not force and self.clock() < start + timedelta(minutes=GRACE_MIN):
                return {"ok": False, "error": f"Grace period ({GRACE_MIN} min after start) has not ended yet."}
            self.db.execute("UPDATE bookings SET status='no_show' WHERE id=?", (rid,))
            n = self.noshow_count(row["phone"])
            extra = f" Future bookings will need a Rs {DEPOSIT_INR} refundable advance." if n >= NOSHOW_LIMIT else ""
            self._log_msg(self.bid(rid), row["phone"], "no_show",
                          f"Hi {row['name'].split()[0]}, we missed you for your {row['service']} on {fmt_dt(start)}. "
                          f"The slot was released. Reply to rebook anytime.{extra}")
            self.db.commit()
            return {"ok": True, "no_shows_for_customer": n, "deposit_now_required": n >= NOSHOW_LIMIT}

    def all_bookings(self):
        return [self._d(r) | {"late_cancel": bool(r["late_cancel"]), "reminded": bool(r["reminded"])}
                for r in self.db.execute("SELECT * FROM bookings ORDER BY start").fetchall()]

    def outbox(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM messages ORDER BY id DESC").fetchall()]

    def handoffs(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM handoffs ORDER BY id DESC").fetchall()]

    # -------------------------------------------------------------------- seed
    def _working_day(self, offset: int):
        d = self.clock() + timedelta(days=offset)
        while d.weekday() == 6:
            d += timedelta(days=1)
        return d.replace(hour=0, minute=0, second=0, microsecond=0)

    def seed(self):
        """Demo data. AUR-1001 / 9876543210 is the sample booking used in the demo video."""
        with self.lock:
            if self.db.execute("SELECT COUNT(*) c FROM bookings").fetchone()["c"]:
                return
            d1, d2 = self._working_day(1), self._working_day(2)
            rows = [
                ("Demo Customer", "9876543210", "Facial", "Neha", d1.replace(hour=15), 60, "booked"),
                ("Seed A", "9000000001", "Haircut", "Aman", d1.replace(hour=11), 30, "booked"),
                ("Seed B", "9000000002", "Haircut", "Aman", d1.replace(hour=11, minute=30), 30, "booked"),
                ("Seed C", "9000000003", "Haircut", "Riya", d1.replace(hour=11), 30, "booked"),
                ("Seed D", "9000000004", "Hair Colour", "Riya", d1.replace(hour=14), 120, "booked"),
                ("Seed E", "9000000005", "Manicure", "Neha", d2.replace(hour=10), 60, "booked"),
                ("Seed F", "9000000006", "Haircut", "Aman", d2.replace(hour=16), 30, "booked"),
                # a repeat no-show customer to demonstrate the deposit rule (phone 9999900000)
                ("Repeat Noshow", "9999900000", "Haircut", "Aman", self.clock() - timedelta(days=9), 30, "no_show"),
                ("Repeat Noshow", "9999900000", "Haircut", "Aman", self.clock() - timedelta(days=3), 30, "no_show"),
            ]
            for n, p, s, st, start, m, status in rows:
                self.db.execute("INSERT INTO bookings(name,phone,service,stylist,start,minutes,status,created_at) VALUES(?,?,?,?,?,?,?,?)",
                                (n, p, s, st, start.strftime("%Y-%m-%dT%H:%M"), m, status, self.clock().isoformat(timespec="minutes")))
            self.db.commit()
