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

- Coach name: `pilot_coach_1`
- Password: `local-only-test-password`

This is only for preview testing. Before opening it to other coaches, add per-coach accounts with unique passwords and avoid sharing the preview account.
