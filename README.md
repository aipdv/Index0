# Better Files

A file manager for Android with meaning search. Find documents and photos by what is in them, not only by their name. Search runs on your phone.

**[Watch the launch video (2:23)](art/better-files-launch.mp4)**

## What it does

- Meaning search for documents and photos. The search index and the model stay on your phone.
- No account.
- Analytics are off until you turn them on. See the [privacy policy](PRIVACY.md).
- File manager: root access, archives, SMB, SFTP, FTP and WebDAV, themes.
- Needs Android 10 or later.

## Install

Download the APK from [GitHub Releases](https://github.com/aipdv/Index0/releases).

The search model (EmbeddingGemma 2, about 320 MB) downloads once, after you accept the Gemma terms. Photo search needs one more download of about 110 MB.

## Known limits

This is a beta.

- The first index takes hours on older phones. You can search files as soon as they are done, newest first.
- Some phones stop apps in the background. On vivo, turn on Settings > Battery > High background power consumption for Better Files. On Xiaomi, Oppo and Realme, allow Autostart and turn off battery limits for the app.
- If indexing stops, open Better Files. It continues where it stopped.
- Phones with less than about 3.5 GB of RAM search inside documents by keyword only. Meaning search and photo search need more memory.
- Scanned PDFs and password-protected PDFs are not searchable yet.
- Excel, PowerPoint and old .doc files are not indexed yet.

## Build

```sh
./gradlew :app:assembleRelease
```

Release builds read the signing key from `signing.properties` (see `signing.properties.example`) or from `STORE_FILE`, `STORE_PASSWORD`, `KEY_ALIAS` and `KEY_PASSWORD`. A build without `POSTHOG_API_KEY` and `POSTHOG_HOST` has no analytics.

## Based on Material Files

Better Files is based on [Material Files](https://github.com/zhanghai/MaterialFiles) by Hai Zhang.

## Changes from Material Files

aipdv changed the program from 2026-10-23. This list was last updated on 2026-10-25.

- New name (Better Files), app id `com.aipdv.bf` and launcher icon.
- Added meaning search and indexing (the Narad engine in `lib/`) and an Indexing screen.
- Removed Firebase Analytics and Crashlytics.
- Added analytics with PostHog EU, off until the user turns them on.
- Minimum Android version is now 10 (API 29).
- About screen and open source notices updated.

## Name, logo and credit

"Better Files" and the Better Files logo are trademarks of aipdv. If you distribute a modified version, use a different name and a different icon.

These additional terms under section 7 of the GPL-3.0 apply to the code that aipdv added to Material Files (the `lib/` search engine and the indexing and search code in `app/`):

- 7(b): Keep the notice "Includes work from Better Files by aipdv (https://github.com/aipdv/Index0)" in the source and on the About or Licenses screen of any app that you build from this code.
- 7(c): Mark modified versions as changed. Do not present them as Better Files or as made by aipdv.
- 7(e): This license gives no rights to the aipdv trademarks above.

## License

    Copyright (C) 2026 aipdv
    Copyright (C) 2018 Hai Zhang

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
