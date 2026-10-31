// Day 22: k6 load test of POST /v2/assess alone (no billing in front), to measure the engine's own capacity.
// Open model: requests start at RATE/s whether or not earlier ones finished. Every request = new customer + new card
// (unique transaction ids, no velocity rules firing), small INR amounts - the same mix billing sends in the checkout test.
//   k6 run -e API_KEY=... -e RATE=20 -e DURATION=1m load-test/assess.js
import http from 'k6/http';
import { check } from 'k6';

const URL = __ENV.RISK_URL || 'http://localhost:8000';
export const options = {
  scenarios: { assess: { executor: 'constant-arrival-rate', rate: Number(__ENV.RATE || 10), timeUnit: '1s',
    duration: __ENV.DURATION || '1m', preAllocatedVUs: 50, maxVUs: 300 } },
  summaryTrendStats: ['avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  // billing gives the whole check 300 ms (ADR 0017): over that, billing uses its fallback rules
  thresholds: { http_req_failed: ['rate<0.01'], http_req_duration: ['p(99)<300'], dropped_iterations: ['count<1'] },
};

const pick = (chars, n) => Array.from({ length: n }, () => chars[Math.random() * chars.length | 0]).join('');
const hex = (n) => pick('0123456789abcdef', n);
const base32 = (n) => pick('abcdefghijklmnopqrstuvwxyz234567', n);   // card-vault token alphabet

export default function () {
  const id = `${exec_id()}`;
  const body = {
    transaction_id: id, amount_minor: 10000 + Math.floor(Math.random() * 190000), currency: 'INR',
    card_token: `tok_${base32(26)}`, card_fingerprint: hex(64), last4: '4242', customer_id: `cust-${hex(12)}`,
    merchant_id: 'hospital-main', ip: '127.0.0.1', device_id: `dev-${hex(12)}`, timestamp: new Date().toISOString(),
  };
  const res = http.post(`${URL}/v2/assess`, JSON.stringify(body),
    { headers: { 'Content-Type': 'application/json', 'X-API-Key': __ENV.API_KEY } });
  check(res, { '200': (r) => r.status === 200 });
}

function exec_id() { return `lt-${__VU}-${__ITER}-${hex(8)}`; }
