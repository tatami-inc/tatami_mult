include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG master # ^v4.1.0
)

FetchContent_Declare(
  sanisizer
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master # ^v0.2.0
)

FetchContent_Declare(
  jiwoo 
  GIT_REPOSITORY https://github.com/LTLA/jiwoo
  GIT_TAG master # ^v0.1.0
)

FetchContent_MakeAvailable(tatami)
FetchContent_MakeAvailable(sanisizer)
FetchContent_MakeAvailable(jiwoo)
