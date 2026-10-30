-- One /v2/assess transaction, atomically (Redis runs a script alone): read what happened BEFORE it, then record it.
-- Two concurrent payments on one card can't both see "0 earlier payments".
--
-- KEYS  1 risk:txn:<id>              the snapshot we returned for this transaction (replay -> same answer, counted once)
--       2 risk:cust:<c>:txns         ZSET txn_id -> ms          customer velocity (the model's txn_count_*)
--       3 risk:cust:<c>:devices      ZSET device -> last ms     devices this customer used (90 days)
--       4 risk:cust:<c>:profile      HASH first_seen, orders, amount_sum (INR minor), chargebacks
--       5 risk:card:<fp>:txns        ZSET txn_id -> ms          card velocity
--       6 risk:card:<fp>:amounts     HASH txn_id -> INR minor   (for the 24h amount sum)
--       7 risk:card:<fp>:devices     ZSET device -> last ms
--       8 risk:card:<fp>:ips         ZSET ip -> last ms
--       9 risk:card:<fp>:first_seen  STRING ms
-- ARGV  1 now (ms)  2 txn_id  3 amount (INR minor)  4 device_id  5 ip
-- Returns {replayed (0/1), snapshot JSON}
local snapshot = redis.call('GET', KEYS[1])
if snapshot then
  return {1, snapshot}
end

local now = tonumber(ARGV[1])
local txn, amount, device, ip = ARGV[2], tonumber(ARGV[3]), ARGV[4], ARGV[5]
local HOUR, DAY = 3600000, 86400000
local KEEP_WINDOW = 25 * 3600              -- seconds: 24h windows + slack
local KEEP_PROFILE = 400 * 86400           -- seconds: profiles outlive a year of silence

-- ---- customer: read (before this txn) ----
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now - DAY)
local cust_1h = redis.call('ZCOUNT', KEYS[2], now - HOUR, '+inf')
local cust_24h = redis.call('ZCARD', KEYS[2])
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now - 90 * DAY)
local device_known = redis.call('ZSCORE', KEYS[3], device)
local profile = redis.call('HMGET', KEYS[4], 'first_seen', 'orders', 'amount_sum', 'chargebacks')

-- ---- card: read (before this txn) ----
local old = redis.call('ZRANGEBYSCORE', KEYS[5], '-inf', now - DAY)
if #old > 0 then
  redis.call('HDEL', KEYS[6], unpack(old))
  redis.call('ZREMRANGEBYSCORE', KEYS[5], '-inf', now - DAY)
end
local card_1h = redis.call('ZCOUNT', KEYS[5], now - HOUR, '+inf')
local card_24h = redis.call('ZCARD', KEYS[5])
local card_sum = 0
for _, v in ipairs(redis.call('HVALS', KEYS[6])) do
  card_sum = card_sum + tonumber(v)
end
redis.call('ZREMRANGEBYSCORE', KEYS[7], '-inf', now - DAY)
redis.call('ZREMRANGEBYSCORE', KEYS[8], '-inf', now - DAY)

-- ---- record this txn ----
redis.call('ZADD', KEYS[2], now, txn)
redis.call('EXPIRE', KEYS[2], KEEP_WINDOW)
redis.call('ZADD', KEYS[3], now, device)
redis.call('EXPIRE', KEYS[3], 90 * 86400)
redis.call('HSETNX', KEYS[4], 'first_seen', now)
redis.call('HINCRBY', KEYS[4], 'orders', 1)
redis.call('HINCRBY', KEYS[4], 'amount_sum', amount)
redis.call('EXPIRE', KEYS[4], KEEP_PROFILE)
redis.call('ZADD', KEYS[5], now, txn)
redis.call('HSET', KEYS[6], txn, amount)
redis.call('ZADD', KEYS[7], now, device)
redis.call('ZADD', KEYS[8], now, ip)
for i = 5, 8 do
  redis.call('EXPIRE', KEYS[i], KEEP_WINDOW)
end
local new_card = redis.call('SET', KEYS[9], now, 'NX', 'EX', KEEP_PROFILE)

snapshot = cjson.encode({
  observed_at_ms = now,
  customer_txn_count_1h = cust_1h,
  customer_txn_count_24h = cust_24h,
  customer_device_known = device_known and 1 or 0,
  customer_first_seen_ms = tonumber(profile[1]) or now,
  customer_prior_orders = tonumber(profile[2]) or 0,
  customer_prior_amount_sum_minor = tonumber(profile[3]) or 0,
  customer_chargebacks = tonumber(profile[4]) or 0,
  card_txn_count_1h = card_1h,
  card_txn_count_24h = card_24h,
  card_amount_sum_24h_minor = card_sum,
  card_distinct_devices_24h = redis.call('ZCARD', KEYS[7]),   -- incl. this one
  card_distinct_ips_24h = redis.call('ZCARD', KEYS[8]),       -- incl. this one
  card_is_new = new_card and 1 or 0
})
redis.call('SET', KEYS[1], snapshot, 'EX', KEEP_WINDOW)
return {0, snapshot}
