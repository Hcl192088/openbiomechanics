# Cloudflare qualitative mechanics preview

This is a minimal Cloudflare deployment target for the qualitative mechanics labeling prototype.

It keeps the local Python prototype separate and reimplements the same core API with:

- Cloudflare Worker for `/api/*`
- Cloudflare D1 for coaches, sessions, tasks, and labels
- Worker static assets for the HTML UI and motion JSON files

## Local preparation

Run these from `D:\baseball\pitching\obp\experiments\qualitative_mechanics\cloudflare`.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\baseball\pitching\obp\baseball_pitching_env\Scripts\python.exe' .\scripts\build_seed_sql.py --write
& 'D:\baseball\pitching\obp\baseball_pitching_env\Scripts\python.exe' .\scripts\prepare_assets.py
npm install
npm run check
```

`seed.sql` and `public/web_motion/` are generated deployment artifacts and are intentionally git-ignored.

If login fails with a PBKDF2 iteration-count error, rebuild `seed.sql` with the current script and rerun both D1 commands below. Cloudflare Workers support up to 100,000 PBKDF2 iterations, so the Cloudflare preview seed intentionally uses 100,000 instead of the local Python prototype's 200,000.

## Cloudflare setup

Create the D1 database:

```powershell
npx wrangler d1 create qualitative-mechanics-labeling
```

Copy the returned `database_id` into `wrangler.toml`.

Apply schema and seed data:

```powershell
npx wrangler d1 execute qualitative-mechanics-labeling --remote --file .\schema.sql
npx wrangler d1 execute qualitative-mechanics-labeling --remote --file .\seed.sql
```

Run a preview:

```powershell
npx wrangler dev
```

Deploy:

```powershell
npx wrangler deploy
```

## Preview account

- Coach name: `hcl`
- Password: `0000`

- Coach name: `ayung`
- Password: `0000`

Both preview accounts are marked as requiring a password reset after login. This is only for preview testing. Before opening it to other coaches, use per-coach accounts with unique passwords and avoid sharing accounts.

## Current security notes

- SQL statements use D1 prepared statements with bound parameters.
- Login and label APIs enforce JSON body, name, password, and notes length limits.
- Session tokens are random bearer tokens stored in D1 and expire after 7 days.
- Password hashes use PBKDF2-SHA256 with 100,000 iterations, which is the Cloudflare Workers WebCrypto limit.
- The app does not yet have brute-force rate limiting, admin account management, or first-class audit logs.
