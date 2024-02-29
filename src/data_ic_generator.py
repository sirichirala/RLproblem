#ic_generator.py file
import logging 
import json 
import pandas as pd 
import typing 
import os
import random
import numpy as np


log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    """Configuration of instance to be generated.

    Args:
        raw_data_path (str): Folder where raw data is located.
        instance_path (str): Folder where file will be created.
        instance_name (str): Name of file to be created. 
        location_id (int): Location id to consider from raw data. 
        lead_time_in_weeks (int): Lead time in weeks
        items_file (str): name of file that contains master list of products. 
        forecast_file (str): name of file that contains the forecast. 
        open_positions_file (str): name of file that contains open positions. 
    """
    raw_data_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    instance_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data'))
    instance_name: str = 'ic_instance'
    items_file: str = 'item_master_50.csv'
    forecast_file: str = 'forecast.csv'
    open_positions_file: str = 'open_po.csv'


class Product(typing.NamedTuple):
    id: int
    on_hand_inventory: int

    def to_json(self): 
        return {
            "id": self.id, 
            "on_hand_inventory": self.on_hand_inventory
        }


class IcData(typing.NamedTuple):
    config: Config
    products: typing.List[Product]

    @classmethod 
    def build(cls, cfg: Config): 
        """ 
        1. Read and filter items in the master item list. 
        2. Collect list of items and create products. 
        """
        # read all the items in the master file 
        items_file = f'{cfg.raw_data_path}/{cfg.items_file}'
        df = pd.read_csv(items_file)
        
        # remove items with zero safety stock and zero on-hand stock
        df.drop(df[(df.on_hand == 0) & (df.safety_stock == 0)].index, inplace = True)

        
        #keep only 5 products (toy problem!!!!!!!)
        #df = df.tail(5)
        

        # create products 
        #products = [Product(id = i, on_hand_inventory = o) 
        #            for i, o in zip(df['item'], df['on_hand'].abs())]
        products = [Product(id=item, on_hand_inventory=random.randint(0, 5)) for item in df['item']]
        product_ids = set([p.id for p in products])
        
        
        return cls(
            config = cfg,
            products = products
        )

    def write_to_file(self, fpath: str):
        cfg = self.config._asdict()
        del cfg['raw_data_path']
        del cfg['instance_path']
        del cfg['items_file'] 
        del cfg['forecast_file'] 
        del cfg['open_positions_file']
        data = {
            "config": cfg,
            "products": [p.to_json() for p in self.products]
        }
        with open(fpath, "w") as outfile:
            json.dump(data, outfile, indent=4)
    
        

def main():
    """Prepare the data to be written to file"""
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.DEBUG)
    cfg = Config()
    log.info('starting data creation.')
    ic_data = IcData.build(cfg)
    fpath = os.path.join(cfg.instance_path, cfg.instance_name + '.json')
    ic_data.write_to_file(fpath)
    log.info(f'wrote instance to {fpath}.')


if __name__ == '__main__':
    main()