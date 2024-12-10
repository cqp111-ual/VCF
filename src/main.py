def main(parser, logging, CoDec):
    #parser.description = __doc__
    args = parser.parse_known_args()[0]
    #args = parser.parse_args()
    print("main", args)

    if args.debug:
        #FORMAT = "%(asctime)s p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
        FORMAT = "(%(levelname)s) %(module)s %(funcName)s %(lineno)d: %(message)s"
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    else:
        FORMAT = "(%(levelname)s) %(module)s: %(message)s"
        logging.basicConfig(format=FORMAT, level=logging.INFO)

    # If parameters "encode" of "decode" are undefined, the following
    # block causes an AttributeError exception.
    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("Sorry, you must specify 'encode' or 'decode'")
        quit()

    # Create an encoder or a decoder, depending on the first
    # parameter.
    codec = CoDec(args)

    # Run the encoder or the decoder
    #BPP, RMSE = args.func(codec)
    #logging.info(f"BPP = {BPP} bits/pixel")
    #logging.info(f"RMSE = {RMSE}")
    #logging.info(f"rate = {args.func(codec)}")
    args.func(codec)
