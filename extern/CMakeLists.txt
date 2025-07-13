include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG master # ^v4.0.0
)

FetchContent_Declare(
  tatami_stats
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_stats
  GIT_TAG master # ^v2.0.0
)

FetchContent_Declare(
  sanisizer
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master # ^v0.1.0
)

FetchContent_MakeAvailable(tatami)
FetchContent_MakeAvailable(tatami_stats)
FetchContent_MakeAvailable(sanisizer)
