import pytest
import os


@pytest.fixture(scope = 'session')
def NEXUSDIR():
    return os.environ['NEXUSDIR']


@pytest.fixture(scope = 'session')
def config_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('configure_tests')


@pytest.fixture(scope = 'session')
def output_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('output_files')


@pytest.fixture(scope = 'session')
def base_name_full_body():
    return 'PET_full_body_sd_test'


@pytest.fixture(scope = 'session')
def nexus_output_file_full_body(output_tmpdir, base_name_full_body):
    n_sipm          = 102304
    n_boards        = 0
    sipms_per_board = 0
    board_ordering  = 0
    return os.path.join(output_tmpdir, base_name_full_body+'.h5'), n_sipm, n_boards, sipms_per_board, board_ordering


@pytest.fixture(scope = 'session')
def base_name_ring_tiles():
    return 'PET_ring_tiles_sd_test'


@pytest.fixture(scope = 'session')
def nexus_output_file_ring_tiles(output_tmpdir, base_name_ring_tiles):
    n_sipm          = 3840
    n_boards        = 120
    sipms_per_board = 32
    board_ordering  = 1000
    return os.path.join(output_tmpdir, base_name_ring_tiles+'.h5'), n_sipm, n_boards, sipms_per_board, board_ordering


@pytest.fixture(scope = 'session')
def base_name_pet_box():
    return 'PET_box_sd_test'


@pytest.fixture(scope = 'session')
def nexus_output_file_pet_box(output_tmpdir, base_name_pet_box):
    n_sipm          = 128
    n_boards        = 0
    sipms_per_board = 0
    board_ordering  = 0
    return os.path.join(output_tmpdir, base_name_pet_box+'.h5'), n_sipm, n_boards, sipms_per_board, board_ordering


@pytest.fixture(scope = 'session')
def base_name_pet_box_HamamatsuVUV():
    return 'PET_box_HamamatsuVUV_sd_test'

@pytest.fixture(scope = 'session')
def base_name_pet_box_HamamatsuBlue():
    return 'PET_box_HamamatsuBlue_sd_test'

@pytest.fixture(scope = 'session')
def base_name_pet_box_FBK():
    return 'PET_box_FBK_sd_test'


@pytest.fixture(scope="module",
         params=["nexus_output_file_full_body", "nexus_output_file_ring_tiles", "nexus_output_file_pet_box"],
         ids=["full_body", "ring_tiles", "pet_box"])
def nexus_files(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module",
         params=["base_name_full_body", "base_name_ring_tiles", "base_name_pet_box"],
         ids=["full_body", "ring_tiles", "pet_box"])
def nexus_filenames(request, output_tmpdir):
    return os.path.join(output_tmpdir, request.getfixturevalue(request.param)+'.h5')
