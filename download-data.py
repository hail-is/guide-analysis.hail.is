import argparse
import base64
import hashlib
import os
import shutil
import tarfile
from datetime import datetime, UTC
from enum import Enum, auto
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from zipfile import ZipFile

import google_crc32c
from google.cloud import storage
# TODO: add logging


class Handler(Enum):
    NONE = auto()
    TAR = auto()
    ZIP = auto()


class FileSpec(NamedTuple):
    name: str
    required: bool


class DataSpec(NamedTuple):
    name: str
    handler: Handler
    required: bool
    provides: list[FileSpec]

    def any_required(self) -> bool:
        return self.required or any(spec.required for spec in self.provides)


DATA = [
    DataSpec(
        'all_phenos.tar.gz',
        Handler.TAR,
        True,
        [FileSpec('all_phenos', True)],
    ),
    DataSpec('betas.npy', Handler.NONE, False, [FileSpec('betas.npy', False)]),
    DataSpec('degas_betas.npy', Handler.NONE, False, [FileSpec('degas_betas.npy', False)]),
    DataSpec(
        'guide_all_100lat_bl_ll.npz',
        Handler.ZIP,
        False,
        [
            FileSpec('guide_all_100lat_bl_ll/contribution_gene.npy', True),
            FileSpec('guide_all_100lat_bl_ll/contribution_phe.npy', True),
            FileSpec('guide_all_100lat_bl_ll/contribution_var.npy', True),
            FileSpec('guide_all_100lat_bl_ll/cos_phe.npy', True),
            FileSpec('guide_all_100lat_bl_ll/cos_var.npy', False),
            FileSpec('guide_all_100lat_bl_ll/factor_phe.npy', True),
            FileSpec('guide_all_100lat_bl_ll/factor_var.npy', True),
            FileSpec('guide_all_100lat_bl_ll/label_gene.npy', False),
            FileSpec('guide_all_100lat_bl_ll/label_phe.npy', True),
            FileSpec('guide_all_100lat_bl_ll/label_phe_code.npy', False),
            FileSpec('guide_all_100lat_bl_ll/label_phe_stackedbar.npy', False),
            FileSpec('guide_all_100lat_bl_ll/label_var.npy', True),
        ],
    ),
    DataSpec('log10pvalues.npy', Handler.NONE, True, [FileSpec('log10pvalues.npy', True)]),
    DataSpec(
        'w_values.npz',
        Handler.ZIP,
        False,
        [
            FileSpec('w_values/label_phe.npy', False),
            FileSpec('w_values/label_var.npy', False),
            FileSpec('w_values/logw_mat_TL.npy', True),
            FileSpec('w_values/logw_mat_XL.npy', True),
            FileSpec('w_values/w_vals_TL.npy', False),
            FileSpec('w_values/w_vals_XL.npy', False),
        ],
    ),
]

for spec in DATA:
    if spec.handler is Handler.NONE:
        if (
            len(spec.provides) != 1
            or spec.provides[0].name != spec.name
            or spec.provides[0].required != spec.required
        ):
            raise ValueError(
                f'Invalid DATA array, {spec}. Either this DataSpec needs a non-NONE '
                'Handler, or the DataSpec and FileSpec must refer to the same single file '
                'and have the same requiredness.'
            )


def compare_checksum(name: str, blob: storage.Blob) -> bool:
    with open(name, 'rb') as fp:
        if (checksum_value := blob.crc32c) is not None:
            cs_ty = 'crc32c'
            checksum = google_crc32c.Checksum()
        elif (checksum_value := blob.md5_hash) is not None:
            cs_ty = 'md5'
            checksum = hashlib.md5()
        else:
            return False

        print(f'checking {name} against cloud version with {cs_ty} checksum')
        for chunk in iter(lambda: fp.read(128 * 1024), b''):
            checksum.update(chunk)

        checksum_b64 = base64.b64encode(checksum.digest()).decode('utf-8')
        matches = checksum_value == checksum_b64
        print(
            f'checksums for {name}{"" if matches else " do not"} match '
            f'[local=`{checksum_value}` cloud=`{checksum_value}`]'
            f'{"" if matches else ": need to refresh"}'
        )
        return matches


def needs_refresh(data_spec: DataSpec, blob: storage.Blob, use_checksums: bool) -> bool:
    # only called when keep_all is false, so we only need to check required files
    #
    # logic here is as follows:
    #   - if a data file's mtime is less than when the blob was last updated, then we
    #     may need to download its provider.
    #   - if we have a provider, then if it's different from the cloud file, we need
    #     to download it.
    #   - in order to check if the provider is different from the cloud file, we first
    #     check to see if their modification times are older
    #   - if they are then we check if they are different
    #      - first by size
    #      - then by checksum
    possibly_needs_refresh = False
    for file_spec in data_spec.provides:
        if not file_spec.required:
            continue
        try:
            stat = os.stat(file_spec.name)
            if datetime.fromtimestamp(stat.st_mtime, tz=UTC) < blob.updated:
                possibly_needs_refresh = True
                break
        except OSError:
            possibly_needs_refresh = True
            break

    if not possibly_needs_refresh:
        return False

    try:
        stat = os.stat(data_spec.name)
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        if mtime < blob.updated:
            if stat.st_size != blob.size:
                print('need to refresh', data_spec.name, 'sizes differ between local and cloud')
                return True
            return use_checksums and not compare_checksum(data_spec.name, blob)
    except OSError:
        pass
    return possibly_needs_refresh


def main(project_id, target_dir, keep_all, use_checksums, bucket_name, path):
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)

    gcs = storage.Client()
    bucket = gcs.bucket(bucket_name, user_project=project_id)

    def get_blob(filename):
        object_name = str(PurePosixPath(path, filename))
        blob = bucket.get_blob(object_name)
        if blob is None:
            raise FileNotFoundError(
                f'gs://{bucket_name}/{object_name} does not exist or '
                'user does not have permission to read the object'
            )
        return blob

    blobs = [(spec, get_blob(spec.name)) for spec in DATA]
    download_file_list = (
        blobs
        if keep_all
        else [
            (spec, blob)
            for spec, blob in blobs
            if spec.any_required() and needs_refresh(spec, blob, use_checksums)
        ]
    )

    refreshed = set()
    for spec, blob in download_file_list:
        print(f'downloading {spec.name}')
        blob.download_to_filename(spec.name)
        refreshed.add(spec.name)

    for spec in DATA:
        extract_if_needed(keep_all, refreshed, spec)


def extract_if_needed(keep_all, refreshed, data_spec):
    if (
        not keep_all
        and data_spec.name not in refreshed
        and all(
            os.path.exists(file_spec.name) for file_spec in data_spec.provides if file_spec.required
        )
    ):
        if data_spec.any_required():
            print('data files fresh for', data_spec.name)
        return

    if data_spec.handler is Handler.NONE:
        print('refreshed', data_spec.name)
        return

    print('extracting data files from', data_spec.name)

    if data_spec.handler is Handler.TAR:
        # the one tarfile is currently transparently a directory, remove it
        # before inflating to ensure fresh files
        # TODO: possibly verbose output
        out_path = Path(data_spec.provides[0].name)
        if out_path.exists():
            shutil.rmtree(out_path)
        with tarfile.open(data_spec.name, 'r:*') as tarf:
            tarf.extractall(filter=tarfile.data_filter)
        out_path.touch()


    if data_spec.handler is Handler.ZIP:
        # TODO: verbose output
        with ZipFile(data_spec.name, mode='r') as zipf:
            # all of our zips are flat archives, this may change, in which case
            # this code would need to be updated to probably just extract the
            # files directly from their name
            paths = [Path(spec.name) for spec in data_spec.provides if keep_all or spec.required]
            [datadir] = set(path.parents[-2] for path in paths)
            for path in paths:
                # print('  x', path.name, '->', path)
                zipf.extract(path.name, path=datadir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='download-data',
        description='Utility for downloading GUIDE browser data from the cloud',
    )
    parser.add_argument(
        'project_id', help='Project for authenticating requester-pays storage requests'
    )
    parser.add_argument(
        'target_dir',
        help='Directory for downloading and unpacking files',
        default=os.getcwd(),
        nargs='?',
    )
    parser.add_argument(
        '--all',
        '-a',
        action='store_true',
        help='download and extract all files, including already downloaded and unneded files')
    parser.add_argument('--use-checksums', action='store_true', help='use checksums in addition to size/mtime to compute file freshness')
    parser.add_argument('--bucket', '-b', help='GCS bucket for data files')
    parser.add_argument('--path-prefix', '-p', help='prefix within storage bucket for data files')

    DEFAULT_BUCKET = 'guide-analysis-browser'
    DEFAULT_PATH = 'data'

    args = parser.parse_args()
    if bool(args.bucket) ^ bool(args.path_prefix):
        parser.error('--bucket and --path-prefix must be specified together')

    main(
        args.project_id,
        args.target_dir,
        args.all,
        args.use_checksums,
        args.bucket or DEFAULT_BUCKET,
        args.path_prefix or DEFAULT_PATH,
    )
