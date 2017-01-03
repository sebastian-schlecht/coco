def make_generator(processor, parallelism=1):
    if parallelism > 0:
        processor.start_daemons(parallelism=parallelism)
    while 1:
        for batch in processor.iterate():
            yield batch
        
        

            
      