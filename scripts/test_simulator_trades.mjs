// Draft-night trades in the simulator page (scripts/simulator_artifact.html), run headless in Node with a stub DOM.
// Build the page first:  uv run python scripts/export_simulator_artifact.py
// Run:  node scripts/test_simulator_trades.mjs [data/derived/artifact/draft_simulator.html]
import fs from "fs";
import vm from "vm";
import assert from "assert";

const file = process.argv[2] || "data/derived/artifact/draft_simulator.html";
const html = fs.readFileSync(file, "utf8");
const js = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const els = {};
const el = id => els[id] ??= { id, innerHTML: "", textContent: "", value: "", hidden: false, disabled: false, querySelectorAll: () => [] };
const ctx = vm.createContext({ document: { getElementById: el }, atob, Blob, Response, DecompressionStream, TextDecoder, Uint8Array,
  Int16Array, Math, JSON, console, setTimeout, localStorage: { s: {}, getItem(k) { return this.s[k] ?? null; }, setItem(k, v) { this.s[k] = v; } } });
vm.runInContext(js, ctx);
for (let i = 0; i < 100 && els.main?.hidden !== false; i++) await new Promise(r => setTimeout(r, 100));
const run = c => vm.runInContext(c, ctx);
const step = () => run(`replay(D.teams[year][team], D.pools[year])`);
const avail = (S, s) => S.next.o.some(x => x.name.includes(s));

// 1. KC 2017: nix the Mahomes trade-up (#27, #91, 2018 R1 for #10)
run(`year="2017"; team="KC"; rule="slot"; actions=[]; render();`);
let S = step();
assert.equal(S.next.t, "trade"); assert.equal(S.next.trade.id, 1196);
run(`act({t:"trade", id:1196, ok:false})`); S = step();
assert.equal(S.next.k, 27);
assert(!avail(S, "Mahomes"), "slot rule: the player really taken at #10 is gone");
assert(avail(S, "Tre'Davious White"), "slot rule: the board at #27 is the real one");
run(`rule="partner"; render();`); S = step();
assert(avail(S, "Mahomes") && !avail(S, "Tre'Davious White"), "partner rule: BUF takes White at #10, Mahomes back on the board");
console.log("KC 2017 nix Mahomes: ok");

// 2. CLE 2014: nix the #4 -> #9 trade-down; the follow-on #9 -> #8 trade is no longer possible
run(`year="2014"; team="CLE"; rule="slot"; actions=[{t:"trade", id:619, ok:false}]; render();`); S = step();
assert.equal(S.next.k, 4); assert(avail(S, "Sammy Watkins"), "trade-down nixed: BUF's #4 player is on the board at #4");
for (let i = 0; i < 40 && S.next; i++) {
  run(S.next.t === "trade" ? `act({t:"trade", id:${S.next.trade.id}, ok:true})` : `act({t:"pick", idx:${S.next.o[0].idx}})`); S = step();
}
assert(S.decided.get(620).auto, "the #9 -> #8 trade can't be made without #9");
console.log("CLE 2014 nix trade-down: ok");

// 3. Every team-year: OK every trade and take the team's real player -> the team's real picks (trade-log gaps aside)
let match = 0, n = 0, errors = 0;
for (const y of Object.keys(run(`D.teams`))) for (const tm of Object.keys(run(`D.teams["${y}"]`))) {
  n++;
  try {
    run(`year="${y}"; team="${tm}"; actions=[]; rule="slot";`);
    S = step();
    for (let i = 0; i < 60 && S.next; i++) {
      if (S.next.t === "trade") run(`act({t:"trade", id:${S.next.trade.id}, ok:true})`);
      else { const tp = S.next.o.find(x => x.at === S.next.k && x.by === tm) || S.next.o[0]; run(`act({t:"pick", idx:${tp.idx}})`); }
      S = step();
    }
    run(`render()`);
    if (S.made.map(m => m.k).join() === run(`D.teams["${y}"]["${tm}"].picks.map(p => p[0]).join()`)) match++;
  } catch (e) { errors++; console.log("error", y, tm, e.message); }
}
console.log(`real drafts reproduced: ${match} of ${n}; errors ${errors}`);
assert.equal(errors, 0); assert(match / n > 0.95, "OK-all replay should reproduce >95% of real drafts");
