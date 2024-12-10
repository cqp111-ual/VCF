import argparse

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

# A way of converting a call to a object's method to a plain function
def encode(codec):
    return codec.encode()

def decode(codec):
    return codec.decode()

class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self, prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True):
        #print(description)
        super().__init__(prog, usage, description, epilog, parents, formatter_class, prefix_chars, fromfile_prefix_chars, argument_default, conflict_handler, add_help, allow_abbrev, exit_on_error)
    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, None)
        #print(self.description)
        #print(__doc__)
        #self.print_help()
        #print("\nThis help only shows information about the top-level parameters.\nTo get information about lower-level parameters, use:\n\"python lower-level_module.py {encode|decode} -h\".\nSee docs/README.md to discover the available modules and their usage level.")
        exit(status)

description = None
with open("/tmp/description.txt", 'r') as f:
    description = f.readline()
        
#def create_parser(description):
# Main parameter of the arguments parser: "encode" or "decode"
parser = CustomArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, exit_on_error=False, description=description)
parser.add_argument("-g", "--debug", action="store_true", help=f"Output debug information")
subparser = parser.add_subparsers(help="You must specify one of the following subcomands:", dest="subparser_name")
parser_encode = subparser.add_parser("encode", help="Encode an image")
parser_decode = subparser.add_parser("decode", help="Decode an image")
parser_encode.set_defaults(func=encode)
parser_decode.set_defaults(func=decode)
#    return parser, parser_encode, parser_decode
