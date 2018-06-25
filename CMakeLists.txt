project(CourseWork)

add_library(tmp tmp.cpp)
add_executable(main main.cpp)
target_link_libraries(main tmp)
