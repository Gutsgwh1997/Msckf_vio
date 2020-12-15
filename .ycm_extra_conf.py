import os
import ycm_core

flags = [
'-Wall',
'-Wextra',
'-Werror',
'-fexceptions',
'-DNDEBUG',
'-std=c++11',
'-x',
'c++',
'-isystem',
'/usr/include',
'-isystem',
'/usr/local/include',
'-isystem',
'/opt/ros/' + os.getenv('ROS_DISTRO') + '/include',
'-isystem',
'/home/guan/catkin_vins/devel/include',
'-isystem',
'/home/guan/catkin_vins/src/my_vins/include'
]
compilation_database_folder = './build'

if os.path.exists(compilation_database_folder):
  database = ycm_core.CompilationDatabase(compilation_database_folder)
else:
  database = None

SOURCE_EXTENSIONS = [ '.cpp', '.cxx', '.cc', '.c', '.m', '.mm' ]

def DirectoryOfThisScript():
  return os.path.dirname( os.path.abspath( __file__ ) )


def IsHeaderFile( filename ):
  extension = os.path.splitext( filename )[ 1 ]
  return extension in [ '.h', '.hxx', '.hpp', '.hh' ]


def GetCompilationInfoForFile( filename ):
  if IsHeaderFile( filename ):
    basename = os.path.splitext( filename )[ 0 ]
    for extension in SOURCE_EXTENSIONS:
      replacement_file = basename + extension
      if os.path.exists( replacement_file ):
        compilation_info = database.GetCompilationInfoForFile(
          replacement_file )
        if compilation_info.compiler_flags_:
          return compilation_info
    return None
  return database.GetCompilationInfoForFile( filename )

def Settings( **kwargs ):
  if not database:
    return {
      'flags': flags,
      'include_paths_relative_to_dir': DirectoryOfThisScript()
    }
  filename = kwargs[ 'filename' ]
  compilation_info = GetCompilationInfoForFile( filename )
  if not compilation_info:
    return None

  return {
    'flags': list( compilation_info.compiler_flags_ ),
    'include_paths_relative_to_dir': compilation_info.compiler_working_dir_
  }
