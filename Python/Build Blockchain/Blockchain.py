'''
Below is a simple implementation of a blockchain called PandasChain. This blockchain stores transactions in 
pandas DataFrames (in-memory) and does not write to disk. The following are the components of this chain:

1. Transaction - A transaction is an exchange of Pandas coins between two parties. In the case of our blockchain, a transaction 
consists of:

    - Sender: The name of the party that is sending i.e. "Bob"
    
    - Receiver: The name of the party that is receiving i.e. "Alice"
    
    - Value: The float amount of Pandas Coins transferred
    
    - Timestamp: The datetime the transaction occured
    
    - Transaction Hash: A SHA-256 hash of the string concatenation of timestamp, sender, receiver and value

2. Block - A block holds a pool of transactions in a DataFrame. The maximum a single block can hold is 10 transactions. 
When a block is created, it contains zero transactions and has a status of UNCOMITTED. Once a block contains 10 transactions, 
that block then is marked COMMITTED and a new block is created for future transactions. Blocks are chained together by 
their block hash ID and previous block hash. Each block, except the first genesis block, tracks the hash of the previous block. 
When a block generates its own hash identifier, it uses the previous blocks hash as one of several strings it will concantenate. 

A block consists of:

    - Sequence ID: A unique sequential number starting at 0 that increments by 1 that identifies each block
    
    - Transactions list: A pandas DataFrame containing all of the transactions contained by the block
    
    - Status: Either UNCOMMITTED or COMMITTED
    
    - Merkle Root: A root hash of transactions. In real blockchains like Bitcoin & Ethereum, a 
    Merkle trie (yes, that's spelled trie!) uses a binary tree. We won't do that here. In our case, we will not use 
    a tree but simply take the hash of the string concatenation of all the transaction hashes 
    in a block once a block is full (reaches 10 transactions)
    
    - Block hash: The hash of this block is created by the hash of the string concatenation of the previous block's 
    hash, the chains hash id, current date time, sequence id of the block and the root Merkle hash. 
    The block hash is generated when a block is full and is committed.

3. PandasChain - A container class that manages all interaction to the internal state of the chain, i.e. users only 
interact with an instance of PandasChain and no other class. A PandasChain class consists of:

    - Name: An arbitrary name of this instance of the chain provided in the constructor when PandasChain is created (see
    test cases for usage examples)
    
    - Chain: A Python list of blocks
    
    - Chain ID: A hash concatenation of a UUID, name of the chain, timestamp of creation of the chain that uniquely
    identifies this chain instance.
    
    - Sequence ID: Tracks the current sequence ID and manages it for new blocks to grab and use
    
    - Previous Hash: Tracks what the previous hash of the just committed block is so that a new block can be instantiated 
    with the previous hash passed into its constructor
    
    - Current block: Which block is current and available to hold incoming transactions

    The only way to interact with a PandasChain instance is via the add_transaction() method that accepts new transactions and 
    methods that print out chain data like display_block_headers(). There should be no other way to reach the underlying
    blocks or pandas DataFrames that hold transactions.

'''


import datetime as dt
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unittest
import uuid


class PandasChain:

    def __init__(self, name):
        self.__name = name.upper()
        self.__chain = []
        self.__id = hashlib.sha256(str(
            str(uuid.uuid4())+self.__name+str(dt.datetime.now())).encode('utf-8')).hexdigest()
        self.__seqid = 0
        self.__prev_hash = None
        self.__current_block = Block(self.__seqid, self.__prev_hash)
        print(self.__name, 'PandasChain created with ID',
              self.__id, 'chain started.')

    # 5 This method loops through all committed and uncommitted blocks and display all transactions in them
    def display_chain(self):
        for x in self.__chain:
            x.display_transactions()
        self.__current_block.display_transactions()

    # This method accepts a new transaction and adds it to current block if block is not full.
    # If block is full, it will delegate the committing and creation of a new current block
    def add_transaction(self, s, r, v):
        if self.__current_block.get_size() >= 10:
            self.__commit_block(self.__current_block)
        self.__current_block.add_transaction(s, r, v)

    # This method is called by add_transaction if a block is full (i.e 10 or more transactions).
    # It is private and therefore not public accessible. It will change the block status to committed, obtain the merkle
    # root hash, generate and set the block's hash, set the prev_hash to the previous block's hash, append this block
    # to the chain list, increment the seq_id and create a new block as the current block
    def __commit_block(self, block):
        # self.__current_block.set_status = "Commited"
        #   self._Block__status = "comitted"
        block.set_status("commited")
        #    self.__current_block.set_status(self._Block__status)
        #   print('{} if blank didnt work'.format(self._Block__status))
        self.__block_hash = hashlib.sha256()
        self.__block_hash.update(('{}{}{}{}{}'.format(self.__prev_hash, self.__id, str(
            dt.datetime.now()), self.__seqid, block.get_simple_merkle_root())).encode('utf-8'))
        self.__block_hash = self.__block_hash.hexdigest()
        block.set_block_hash(self.__block_hash)
        self.__prev_hash = self.__block_hash
        self.__chain.append(self.__current_block)
        self.__seqid = self.__seqid+1
        self.__current_block = Block(self.__seqid, self.__prev_hash)
        print('Block committed')

    # Display just the metadata of all blocks (committed or uncommitted), one block per line.
    def display_block_headers(self):
        for each_block in self.__chain:
            each_block.display_header()    
        self.__current_block.display_header()  #                                                                                                                                            self.__current_block._Block__prev_hash,   self.__current_block.get_simple_merkle_root(),   self.__current_block.get_size() ))
      
    #return int total number of blocks in this chain (committed and uncommitted blocks combined
    def get_number_of_blocks(self):
        return(self.__seqid+1)
  

    #Returns all of the values (Pandas coins transferred) of all transactions from every block as a single list
    def get_values(self):
        coin_list = []
        time_value_list=[]
        for each_block in self.__chain:
            block_values,timestamp_values = each_block.get_values()
            for x in block_values:
                coin_list.append(x)
            for x in timestamp_values:
                time_value_list.append(x)
        current_block_vals, current_timestampvals = self.__current_block.get_values()
          
        for x in current_block_vals:
                coin_list.append(x)
        for x in current_timestampvals:
                time_value_list.append(x)
      #  dates = plt.dates.date2num(current_timestampvals)
        return ([time_value_list,coin_list])


class Block:
    # 5 pts for constructor
    def __init__(self, seq_id, prev_hash):
        self.__seq_id = seq_id
        self.__prev_hash = prev_hash
        self.__col_names = ['Timestamp', 'Sender',
                            'Receiver', 'Value', 'TxHash']
        self.__transactions = pd.DataFrame(columns=self.__col_names)
        self.__status = "UNCOMMITTED"
        self.__block_hash = None
        self.__merkle_tx_hash = None

    # Display on a single line the metadata of this block. You'll display the sequence Id, status,
    # block hash, previous block's hash, merkle hash and number of transactions in the block
    def display_header(self):
        print('seqid : {}  Block status : {}   Block hash : {}    Prev Hash : {} ,  merkleroot : {} "  block size : "{}'.format(self.__seq_id, self.__status, self.__block_hash, self.__prev_hash, self.get_simple_merkle_root(), self.__transactions.shape[0]))

    # This is the interface for how transactions are added
    def add_transaction(self, s, r, v):
        ts = dt.datetime.now()
        tx_hash = hashlib.sha256()
        tx_hash.update(('{}{}{}{}'.format(ts, s, r, v)).encode('utf-8'))
        tx_hash = tx_hash.hexdigest()
        new_transaction = pd.DataFrame([[ts, s, r, v, tx_hash]], columns=self.__col_names)
        self.__transactions = self.__transactions.append(new_transaction)

    # Print all transactions contained by this block
    def display_transactions(self):
        print(self.__transactions)

    # Return the number of transactions contained by this block
    def get_size(self):
        return(self.__transactions.shape[0])

    # Setter for status - Allow for the change of status (only two statuses exist - COMMITTED or UNCOMMITTED).
    def set_status(self, status):
        self.__status = status

    # Setter for block hash
    def set_block_hash(self, hash):
        self.__block_hash = hash

    # Return and calculate merkle hash by taking all transaction hashes, concatenate them into one string and
    # hash that string producing a "merkle root" - Note, this is not how merkle tries work but is instructive
    # and indicative in terms of the intent and purpose of merkle tries
    def get_simple_merkle_root(self):
        merkle_string = self.__transactions["TxHash"].sum()
        merk_root = hashlib.sha256()
        merk_root.update(('{}'.format(merkle_string)).encode('utf-8'))
        merk_root = merk_root.hexdigest()
        self.__merkle_tx_hash = merk_root
        return (self.__merkle_tx_hash)
        
    ## returns tx values and timesstamp
    def get_values(self):
        return([self.__transactions["Value"],self.__transactions["Timestamp"]])

if __name__ == '__main__':
    unittest.main()
