include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG master # ^v3.0.0
)

FetchContent_Declare(
  tatami_stats
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_stats
  GIT_TAG master # ^v1.1.0
)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master # ^v0.3.2
)

FetchContent_MakeAvailable(tatami)
FetchContent_MakeAvailable(tatami_stats)
FetchContent_MakeAvailable(subpar)
