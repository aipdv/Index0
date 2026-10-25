# Better Files

A file manager for Android with meaning search. Find documents and photos by what is in them, not only by their name. Search runs on your phone.

## What it does

- Meaning search for documents and photos. The search index and the model stay on your phone.
- No account.
- Analytics are off until you turn them on. See the [privacy policy](PRIVACY.md).
- File manager features from Material Files: root access, archives, SMB, SFTP, FTP and WebDAV, themes.
- Needs Android 10 or later.

## Install

Download the APK from [GitHub Releases](https://github.com/aipdv/Index0/releases).

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
