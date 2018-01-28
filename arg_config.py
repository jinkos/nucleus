from __future__ import print_function

import configparser
import argparse
import sys

# global variables so that args and config can be easily accessed
cfg = None
args = None

# eg...
# arg_config.args.load is True of the --load flag was added
# arg_config.args.machine is the first argument

# There should be a different config setup for each machine
# and each machine should have its own name.
# The name of the machine should be passed as the first argument
def _do_config(config_fname,section,do_print=False):
  global cfg
  
  # the config file stores a number of different configurations
  # 'my_mac' and 'topsecret.server.com'
  
  config = configparser.ConfigParser()
  config.read(config_fname)

  if do_print:  
    print('do_config()...')
    print('sections:',config.sections())
    print('seleted sections:',section)
    for key in config[section]: print(key,config[section][key])
    print()
  
  cfg = config[section]
  
def arg_config(do_print):
  global args
  
  num_args = len(sys.argv)

  if do_print:  
    print('arg_config()...')
    print("num args:",num_args)

  # always...
  parser = argparse.ArgumentParser(description="nucleus")

  # first positional argument
  parser.add_argument("machine", help="my_mac or my_linux")

  # optional arguments
  parser.add_argument("--load", help="load model",action="store_true")
  parser.add_argument("--save", help="save model",action="store_true")
  parser.add_argument("--val", help="validate model",action="store_true")
  
  # always...
  args = parser.parse_args()

  if do_print:  
    print("using machine:",args.machine)
    print()
      
  _do_config("arg_config.cfg",args.machine,do_print)
