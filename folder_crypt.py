#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "cryptography>=42.0.0",
# ]
# ///
"""
folder_crypt.py - Recursive folder encryption / decryption (AES-256-GCM + Scrypt)

Design summary
--------------
* Key derivation : Scrypt (memory-hard) with a random 16-byte salt per run.
* Encryption     : AES-256-GCM, authenticated, streamed in 1 MiB chunks so
                   large files never need to fit in memory.
* Nonces         : every file gets its own random 7-byte nonce prefix; each
                   12-byte GCM nonce = prefix || 4-byte chunk counter ||
                   1-byte "last chunk" flag.  A nonce is never reused with the
                   same key, and truncation / chunk re-ordering is detected.
* Header         : (magic, version, KDF params, chunk size, salt, nonce prefix)
                   is authenticated as Associated Data in every chunk, so
                   tampering with it makes decryption fail.
* Safety         : originals are NEVER modified, overwritten or deleted.
                   Output goes into a NEW sibling folder that mirrors the
                   source structure. Existing files are never overwritten
                   (files are created with exclusive mode "xb").
                   Symlinks / junctions / special files are skipped, and
                   well-known system folders are refused.

File format (".encrypted")
--------------------------
    offset  size  field
    0       4     magic  b"FENC"
    4       1     format version (1)
    5       1     log2(N) Scrypt cost
    6       1     r       Scrypt block size
    7       1     p       Scrypt parallelism
    8       4     chunk size in bytes (big endian)
    12      16    salt
    28      7     nonce prefix
    35      ...   sequence of AES-GCM chunks (ciphertext + 16-byte tag)
"""

import os
import stat
import struct
import sys
import time
import getpass
import secrets
from contextlib import nullcontext
from pathlib import Path

try:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency. Install it with:  pip install -r requirements.txt")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EXT = ".encrypted"
MAGIC = b"FENC"
VERSION = 1

SALT_LEN = 16
PREFIX_LEN = 7            # random per-file part of the 12-byte GCM nonce
TAG_LEN = 16              # GCM authentication tag size
CHUNK_SIZE = 1024 * 1024  # 1 MiB plaintext per chunk
MAX_CHUNK_SIZE = 64 * 1024 * 1024  # sanity limit when reading headers

# Scrypt parameters: N = 2**15, r = 8, p = 1  (~32 MiB RAM, ~0.1 s)
SCRYPT_LOG2N = 15
SCRYPT_R = 8
SCRYPT_P = 1
# Upper bounds accepted when *reading* a file (protects against malicious
# headers that would force enormous memory/CPU use).
MAX_LOG2N, MAX_R, MAX_P = 20, 16, 4

HEADER_STRUCT = struct.Struct(">4sBBBBI")
HEADER_LEN = HEADER_STRUCT.size + SALT_LEN + PREFIX_LEN  # 35 bytes

MIN_PASSWORD_LEN = 8
EARLY_ABORT_AFTER = 3     # abort decryption if the first N files all fail auth


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class CryptoFileError(Exception):
    """Base class for per-file problems (reported, never fatal to the run)."""


class BadPasswordOrCorrupt(CryptoFileError):
    """First chunk failed authentication: wrong password or damaged header."""


class CorruptedFileError(CryptoFileError):
    """File is damaged, truncated, tampered with, or not in our format."""


# --------------------------------------------------------------------------- #
# Cryptographic helpers
# --------------------------------------------------------------------------- #
def derive_key(password: str, salt: bytes, log2n: int, r: int, p: int) -> bytes:
    """Derive a 32-byte (AES-256) key from the password using Scrypt."""
    kdf = Scrypt(salt=salt, length=32, n=2 ** log2n, r=r, p=p)
    return kdf.derive(password.encode("utf-8"))


def make_nonce(prefix: bytes, counter: int, is_last: bool) -> bytes:
    """
    Build a 12-byte GCM nonce: 7-byte random prefix + 4-byte counter + 1 flag.
    The 'last' flag makes truncated files fail authentication.
    """
    if counter >= 2 ** 32:
        raise CryptoFileError("File too large for this format.")
    return prefix + counter.to_bytes(4, "big") + (b"\x01" if is_last else b"\x00")


def build_header(salt: bytes, prefix: bytes) -> bytes:
    return HEADER_STRUCT.pack(
        MAGIC, VERSION, SCRYPT_LOG2N, SCRYPT_R, SCRYPT_P, CHUNK_SIZE
    ) + salt + prefix


def parse_header(header: bytes):
    """Validate and unpack a header. Returns (log2n, r, p, chunk, salt, prefix)."""
    if len(header) != HEADER_LEN:
        raise CorruptedFileError("File too short / not an encrypted file.")
    magic, version, log2n, r, p, chunk = HEADER_STRUCT.unpack(
        header[:HEADER_STRUCT.size]
    )
    if magic != MAGIC:
        raise CorruptedFileError("Not a file produced by this program.")
    if version != VERSION:
        raise CorruptedFileError(f"Unsupported format version {version}.")
    if not (1 <= log2n <= MAX_LOG2N and 1 <= r <= MAX_R and 1 <= p <= MAX_P):
        raise CorruptedFileError("Invalid key-derivation parameters in header.")
    if not (1 <= chunk <= MAX_CHUNK_SIZE):
        raise CorruptedFileError("Invalid chunk size in header.")
    salt = header[HEADER_STRUCT.size:HEADER_STRUCT.size + SALT_LEN]
    prefix = header[HEADER_STRUCT.size + SALT_LEN:]
    return log2n, r, p, chunk, salt, prefix


# --------------------------------------------------------------------------- #
# Single-file operations
# --------------------------------------------------------------------------- #
def _remove_quietly(path: Path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def encrypt_file(src: Path, dst: Path, key: bytes, salt: bytes) -> None:
    """
    Encrypt `src` into the NEW file `dst` (never overwrites: mode 'xb').
    On any failure the partially written `dst` is removed.
    """
    aes = AESGCM(key)
    prefix = secrets.token_bytes(PREFIX_LEN)   # unique nonce prefix per file
    header = build_header(salt, prefix)

    fout = open(dst, "xb")  # raises FileExistsError if dst exists -> no overwrite
    try:
        with open(src, "rb") as fin, fout:
            fout.write(header)
            counter = 0
            chunk = fin.read(CHUNK_SIZE)
            while True:
                # Look ahead so we know whether this is the final chunk.
                nxt = fin.read(CHUNK_SIZE) if len(chunk) == CHUNK_SIZE else b""
                is_last = len(nxt) == 0
                nonce = make_nonce(prefix, counter, is_last)
                fout.write(aes.encrypt(nonce, chunk, header))  # header = AAD
                if is_last:
                    break
                chunk = nxt
                counter += 1
    except BaseException:
        _remove_quietly(dst)
        raise


def decrypt_file(src: Path, dst, password: str, key_cache: dict) -> None:
    """
    Decrypt `src` into the NEW file `dst`. Raises BadPasswordOrCorrupt /
    CorruptedFileError on authentication problems; partial output is removed.
    `key_cache` avoids re-running Scrypt for files sharing the same salt.
    If `dst` is None nothing is written: the file is only verified.
    """
    with open(src, "rb") as fin:
        header = fin.read(HEADER_LEN)
        log2n, r, p, chunk_size, salt, prefix = parse_header(header)

        cache_key = (salt, log2n, r, p)
        if cache_key not in key_cache:
            key_cache[cache_key] = derive_key(password, salt, log2n, r, p)
        aes = AESGCM(key_cache[cache_key])

        enc_size = chunk_size + TAG_LEN
        fout = open(dst, "xb") if dst is not None else None  # never overwrite
        try:
            with (fout if fout is not None else nullcontext()):
                counter = 0
                chunk = fin.read(enc_size)
                if len(chunk) < TAG_LEN:
                    raise CorruptedFileError("File is truncated.")
                while True:
                    nxt = fin.read(enc_size) if len(chunk) == enc_size else b""
                    is_last = len(nxt) == 0
                    nonce = make_nonce(prefix, counter, is_last)
                    try:
                        plain = aes.decrypt(nonce, chunk, header)
                    except InvalidTag:
                        if counter == 0:
                            raise BadPasswordOrCorrupt(
                                "Wrong password or corrupted file."
                            ) from None
                        raise CorruptedFileError(
                            "Data is corrupted, truncated or tampered with."
                        ) from None
                    if fout is not None:
                        fout.write(plain)
                    if is_last:
                        break
                    chunk = nxt
                    counter += 1
        except BaseException:
            if dst is not None:
                _remove_quietly(dst)
            raise


# --------------------------------------------------------------------------- #
# Safety checks and folder scanning
# --------------------------------------------------------------------------- #
def _protected_paths() -> list:
    """System locations that must never be processed (per platform)."""
    paths = []
    if os.name == "nt":
        for var in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)",
                    "ProgramData"):
            val = os.environ.get(var)
            if val:
                paths.append(val)
    else:
        paths += ["/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64",
                  "/proc", "/sbin", "/sys", "/usr", "/run", "/root", "/snap",
                  "/System", "/Library", "/Applications", "/private/etc",
                  "/var/lib", "/var/log", "/var/spool"]
    resolved = []
    for p in paths:
        try:
            resolved.append(Path(p).resolve())
        except OSError:
            pass
    return resolved


def is_protected_path(path: Path) -> bool:
    """True if `path` is a filesystem root, the home folder, or inside a system folder."""
    p = path.resolve()
    if p == Path(p.anchor):                       # "/" or "C:\"
        return True
    if p == Path.home().resolve():                # whole home folder
        return True
    return any(p == q or q in p.parents for q in _protected_paths())


def is_link_like(path: str) -> bool:
    """True for symlinks (and Windows junctions where detectable)."""
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    return bool(isjunction and isjunction(path))


def scan_folder(root: Path):
    """
    Walk `root` recursively WITHOUT following links.
    Returns (files, dirs, skipped, errors):
        files   - relative paths of regular files
        dirs    - relative paths of subdirectories
        skipped - list of (relative path, reason)
        errors  - list of (path, message) for unreadable directories
    """
    files, dirs, skipped, errors = [], [], [], []

    def on_error(err):
        errors.append((str(err.filename), str(err.strerror)))

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=on_error):
        keep = []
        for d in sorted(dirnames):
            full = os.path.join(dirpath, d)
            rel = os.path.relpath(full, root)
            if is_link_like(full):
                skipped.append((rel, "symbolic link / junction"))
            else:
                keep.append(d)
                dirs.append(Path(rel))
        dirnames[:] = keep  # prune so os.walk never enters skipped dirs

        for f in sorted(filenames):
            full = os.path.join(dirpath, f)
            rel = Path(os.path.relpath(full, root))
            if is_link_like(full):
                skipped.append((str(rel), "symbolic link"))
                continue
            try:
                if not stat.S_ISREG(os.lstat(full).st_mode):
                    skipped.append((str(rel), "not a regular file"))
                    continue
            except OSError as exc:
                errors.append((str(rel), str(exc)))
                continue
            files.append(rel)
    return files, dirs, skipped, errors


def unique_output_dir(source: Path, suffix: str) -> Path:
    """Pick '<source>_<suffix>' next to the source, adding a counter if taken."""
    base = source.parent / f"{source.name}_{suffix}"
    candidate, n = base, 1
    while candidate.exists():
        n += 1
        candidate = source.parent / f"{base.name}_{n}"
    return candidate


def validate_output_dir(source: Path, out: Path) -> None:
    """Raise ValueError when the chosen output folder is unsafe."""
    out = out.resolve()
    if is_protected_path(out):
        raise ValueError("Output folder is a protected system location.")
    if out == source or source in out.parents:
        raise ValueError("Output folder must not be inside the source folder.")
    if out in source.parents:
        raise ValueError("Output folder must not contain the source folder.")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output folder already exists and is not empty "
                         "(nothing is ever overwritten).")


# --------------------------------------------------------------------------- #
# Folder-level processing
# --------------------------------------------------------------------------- #
def process_folder(mode: str, source: Path, out_dir: Path, password: str,
                   in_place: bool = False) -> dict:
    """
    Encrypt ('encrypt') or decrypt ('decrypt') every eligible file in `source`,
    writing results into `out_dir` with the same relative structure.
    Errors on one file never stop the others. Returns a report dict.

    in_place=True: `out_dir` must equal `source`. Each file is replaced by its
    encrypted/decrypted version, but the old file is deleted ONLY after the new
    one has been written and verified (GCM-authenticated), so a failure for any
    file leaves that file exactly as it was.
    """
    encrypting = mode == "encrypt"
    all_files, dirs, skipped, errors = scan_folder(source)

    # Choose which files take part.
    targets = []
    for rel in all_files:
        is_enc = rel.name.lower().endswith(EXT)
        if encrypting and is_enc:
            skipped.append((str(rel), "already has .encrypted extension"))
        elif not encrypting and not is_enc:
            skipped.append((str(rel), "not a .encrypted file"))
        else:
            targets.append(rel)

    report = {"total": len(targets), "ok": 0, "failed": [], "skipped": skipped,
              "errors": errors, "out_dir": out_dir, "aborted": None,
              "in_place": in_place}
    if not targets:
        return report

    if not in_place:
        out_dir.mkdir(parents=True, exist_ok=True)
    for d in dirs:                                   # mirror folder structure
        try:
            (out_dir / d).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass                                      # created on demand later

    # One random salt + one key per encryption run (Scrypt is deliberately slow).
    salt = secrets.token_bytes(SALT_LEN)
    key = derive_key(password, salt, SCRYPT_LOG2N, SCRYPT_R, SCRYPT_P) if encrypting else None
    key_cache: dict = {}
    if encrypting:  # lets us verify freshly written files without a 2nd Scrypt run
        key_cache[(salt, SCRYPT_LOG2N, SCRYPT_R, SCRYPT_P)] = key
    auth_failures_in_a_row = 0

    for i, rel in enumerate(targets, 1):
        src = source / rel
        if encrypting:
            dst = out_dir / rel.parent / (rel.name + EXT)
        else:
            new_name = rel.name[:-len(EXT)] or "decrypted_file"
            dst = out_dir / rel.parent / new_name
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if encrypting:
                encrypt_file(src, dst, key, salt)
                if in_place:
                    # Prove the new file decrypts before touching the original.
                    try:
                        decrypt_file(dst, None, password, key_cache)
                    except BaseException:
                        _remove_quietly(dst)
                        raise
            else:
                decrypt_file(src, dst, password, key_cache)  # authenticated as it writes
            if in_place:
                try:
                    os.remove(src)   # only reached when the new file is good
                except OSError as exc:
                    raise CryptoFileError(
                        f"done, but could not remove the old file: {exc}") from None
            report["ok"] += 1
            auth_failures_in_a_row = 0
            print(f"[{i}/{len(targets)}] OK      {rel}")
        except FileExistsError:
            report["failed"].append((str(rel), "destination already exists (not overwritten)"))
            print(f"[{i}/{len(targets)}] FAILED  {rel}  (destination exists)")
        except BadPasswordOrCorrupt as exc:
            report["failed"].append((str(rel), str(exc)))
            print(f"[{i}/{len(targets)}] FAILED  {rel}  ({exc})")
            auth_failures_in_a_row += 1
            if report["ok"] == 0 and auth_failures_in_a_row >= EARLY_ABORT_AFTER:
                report["aborted"] = ("The first files all failed authentication - the "
                                     "password is almost certainly wrong. Stopped early.")
                break
        except (CryptoFileError, OSError) as exc:
            report["failed"].append((str(rel), str(exc)))
            print(f"[{i}/{len(targets)}] FAILED  {rel}  ({exc})")
        except Exception as exc:  # unexpected - still keep going
            report["failed"].append((str(rel), f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(targets)}] FAILED  {rel}  ({type(exc).__name__}: {exc})")

    return report


def print_report(mode: str, report: dict, seconds: float) -> None:
    """Show the final summary."""
    processed = report["ok"] + len(report["failed"])
    print("\n" + "=" * 60)
    print(f" {mode.upper()} REPORT")
    print("=" * 60)
    print(f" Files found to process : {report['total']}")
    print(f" Processed              : {processed}")
    print(f" Successful             : {report['ok']}")
    print(f" Failed                 : {len(report['failed'])}")
    print(f" Skipped                : {len(report['skipped'])}")
    print(f" Time                   : {seconds:.2f} s")
    print(f" Output folder          : {report['out_dir']}"
          + ("  (in place)" if report.get("in_place") else ""))
    if report["aborted"]:
        print(f"\n ! {report['aborted']}")
    if report["failed"]:
        print("\n Failed files:")
        for name, why in report["failed"]:
            print(f"   - {name}: {why}")
    if report["skipped"]:
        print("\n Skipped items:")
        for name, why in report["skipped"]:
            print(f"   - {name}: {why}")
    if report["errors"]:
        print("\n Unreadable locations:")
        for name, why in report["errors"]:
            print(f"   - {name}: {why}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# User interface
# --------------------------------------------------------------------------- #
def ask_folder(prompt: str) -> Path:
    """Ask for an existing folder; returns a resolved Path or None."""
    raw = input(prompt).strip().strip('"').strip("'")
    if not raw:
        return None
    path = Path(os.path.expanduser(raw))
    if not path.is_dir():
        print("  ! That is not an existing folder.")
        return None
    if is_link_like(str(path)):
        print("  ! The selected folder is a symbolic link; refusing.")
        return None
    path = path.resolve()
    if is_protected_path(path):
        print("  ! Refusing to work on a system folder, a drive root or your whole home folder.")
        return None
    return path


def ask_password(confirm: bool) -> str:
    """Read a password without echo; optionally require confirmation."""
    pw = getpass.getpass("Password (hidden): ")
    if confirm:
        if len(pw) < MIN_PASSWORD_LEN:
            print(f"  ! Password must be at least {MIN_PASSWORD_LEN} characters.")
            return None
        if getpass.getpass("Repeat password:   ") != pw:
            print("  ! Passwords do not match.")
            return None
        print("  NOTE: if you forget this password your data CANNOT be recovered.")
    elif not pw:
        print("  ! Password cannot be empty.")
        return None
    return pw


def run_mode(mode: str) -> None:
    """Interactive flow shared by encrypt and decrypt."""
    encrypting = mode == "encrypt"
    source = ask_folder(f"Folder to {mode}: ")
    if source is None:
        return

    ans = input("Work IN PLACE on this same folder? [Y/n] "
                "(n = write to a separate copy): ").strip().lower()
    in_place = ans in ("", "y", "yes")

    if in_place:
        out_dir = source
        action = "encrypted (-> name.ext.encrypted)" if encrypting else "decrypted (.encrypted removed)"
        print(f"\n  Files inside '{source}' will be {action}.")
        print("  Each old file is deleted ONLY after the new one is written and verified.")
        print("  Tip: keep a backup of important data first.")
        if input("  Type YES to continue: ").strip() != "YES":
            print("  Cancelled.")
            return
    else:
        default_out = unique_output_dir(source, "encrypted" if encrypting else "decrypted")
        raw = input(f"Output folder [{default_out}]: ").strip().strip('"').strip("'")
        out_dir = Path(os.path.expanduser(raw)) if raw else default_out
        try:
            validate_output_dir(source, out_dir)
        except ValueError as exc:
            print(f"  ! {exc}")
            return
        out_dir = out_dir.resolve()

    password = ask_password(confirm=encrypting)
    if password is None:
        return

    if in_place:
        print(f"\nProcessing '{source}' in place...\n")
    else:
        print(f"\nOriginals in '{source}' will NOT be modified.")
        print(f"Results will be written to '{out_dir}'.\n")
    start = time.time()
    try:
        report = process_folder(mode, source, out_dir, password, in_place)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return
    except Exception as exc:
        print(f"\nFatal error: {type(exc).__name__}: {exc}")
        return
    print_report(mode, report, time.time() - start)


def main() -> None:
    menu = ("\n=== Folder Encryptor (AES-256-GCM + Scrypt) ===\n"
            "  1) Encrypt a folder\n"
            "  2) Decrypt a folder\n"
            "  3) Exit\n")
    while True:
        print(menu)
        try:
            choice = input("Select an option [1-3]: ").strip()
            if choice == "1":
                run_mode("encrypt")
            elif choice == "2":
                run_mode("decrypt")
            elif choice == "3":
                print("Goodbye.")
                return
            else:
                print("Please enter 1, 2 or 3.")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return


if __name__ == "__main__":
    main()
