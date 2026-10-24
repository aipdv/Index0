# Privacy Policy

Last updated: 2026-10-25.

Index0 is a file manager with on-device search. It has no account and no servers of its own.

## Who runs Index0

Index0 is made by aipdv, an individual developer. To contact us, open an issue on the project's GitHub page.

## Summary

- Analytics are **off** until you turn them on. Nothing is sent before you agree.
- If you turn them on, Index0 sends anonymous usage counts to PostHog in the EU.
- Your files, file names, folders, and searches never leave your phone.
- No ads, no advertising ID, no crash reporting, no account.

## Analytics (only if you opt in)

Index0 asks once, after you have used the app a few times. You can also turn analytics on or off at any time in Settings > Privacy. If you say "No thanks" or close the question, analytics stay off.

Before you opt in, Index0 does not start the analytics library, stores no analytics ID, and sends no analytics data.

When you opt in, Index0 creates a random anonymous ID and stores it on your phone. It is not linked to your name, your account, or your device ID.

### Events we collect

| Event | Extra data |
|---|---|
| App installed, updated, or opened | App version and build; previous version on update; whether it came from the background |
| Files queued for delete or archive | Number of files |
| Files queued for copy or move | Number of files; copy or move |
| File renamed, bookmark added | None |
| Files shared | Number of files |
| File or folder created | File or folder |
| Meaning search used | None |
| Indexing started, re-index started | None |
| Search model download started, completed, or failed | None |
| Overnight indexing turned on or off | On or off |
| Network storage added | Type only: SMB, SFTP, FTP, or WebDAV |

Every event also carries: the anonymous ID, the time, the app version and build, the Android version, the analytics library name and version, and whether the phone is an emulator.

### What we never collect

- File or folder names, paths, or file content
- Search queries, search results, or snippets
- Your search index
- Server addresses, user names, or passwords for network storage
- Your IP address (PostHog discards it) or your location
- Advertising ID or Android ID
- Device make, model, screen size, language, time zone, or mobile carrier
- Screen views, taps, session recordings, crash reports, or logs

### Why and on what legal basis

We use the counts to learn which features people use and which app versions are in use, so we can fix and improve the app. The legal basis is your consent (GDPR Article 6(1)(a)). You can withdraw it at any time.

### Who processes the data

PostHog Inc. processes the analytics data for us on PostHog's EU cloud (servers in Frankfurt, Germany). PostHog acts as our data processor.

### How long we keep it

Analytics events are deleted after 30 days.

### Your choices and rights

- **Turn it off:** Settings > Privacy. Index0 stops recording events and replaces the anonymous ID, so later use cannot be linked to earlier events. Events recorded just before you turned it off may still be sent in the next batch.
- **Access or deletion:** The data has no name or account attached, and Index0 does not show the ID, so we usually cannot find your events. Open an issue if you have a question, and we will help where we can.
- You can also complain to your data protection authority.

## Network use

Index0 connects to the internet only for these features:

- **Search model download (optional).** Meaning search needs a model. You agree to the Gemma terms first. Then Index0 downloads the model from Hugging Face. Hugging Face sees your IP address, like any download. The model then runs on your phone.
- **Network storage and FTP server.** Index0 connects to SMB, SFTP, FTP, and WebDAV servers only when you add them. The FTP server runs only when you start it.
- **Analytics (only if you opt in).** Index0 sends the events above to PostHog EU.

Builds made without an analytics key (for example, builds from source) have no analytics at all. The Settings > Privacy option does not show in those builds.

## Data on your phone

- The search index and the downloaded models stay on your phone. They are excluded from Android backup.
- App settings can be included in Android backup. These settings include saved network storage logins and your analytics choice.

## Children

Index0 is not directed at children. We do not knowingly collect data from children.

## Changes to this policy

When this policy changes, we update the date at the top and mention the change in the release notes.

## Contact

Open an issue on the project's GitHub page.
