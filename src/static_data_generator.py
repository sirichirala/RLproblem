# reads raw data, processes it and generates demand scenarios for each product 

import logging 
import json 
import pandas as pd 
import typing 
import os
import math
import json

import numpy as np
import scipy.stats as stats

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    """Configuration of instance to be generated.

    Args:
        raw_data_path (str): Folder where raw data is located.
        instance_path (str): Folder where file will be created.
        num_time_points (int): Number of time points for which demands have to be created.
        instance_name (str): Name of file to be created. 
        location_id (int): Location id to consider from raw data. 
        lead_time_in_weeks (int): Lead time in weeks
        container_volume (float): Volume of each container 
        items_file (str): name of file that contains master list of products. 
        forecast_file (str): name of file that contains the forecast. 
        open_positions_file (str): name of file that contains open positions. 
    """
    raw_data_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    instance_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data'))
    num_time_points: int = 312 
    instance_name: str = 'static_instance'
    location_id: int = 3000
    lead_time_in_weeks: int = 12 
    container_volume: float = 2350.0
    items_file: str = 'item_master.csv'
    forecast_file: str = 'forecast.csv'
    open_positions_file: str = 'open_po.csv'

class Product(typing.NamedTuple):
    id: int 
    volume: float
    safety_stock_in_weeks: int
    
    def to_json(self): 
        return {
            "id": self.id, 
            "volume": self.volume, 
            "safety_stock_in_weeks": self.safety_stock_in_weeks
        }

class StaticData(typing.NamedTuple):
    config: Config
    products: typing.List[Product]
    product_mle_params: typing.Dict[int, typing.Dict[str, float]]
    
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
        
        # create products 
        products = [Product(id = i, volume = v, safety_stock_in_weeks = s) 
                    for i, v, s in zip(df['item'], df['volume'], df['safety_stock'])]
        product_ids = set([p.id for p in products])
        
        """ 
        1. Read demand forecast for each product, and remove leading/trailing zeros
        2. Fit a gamma distribution for the forecast
        """ 
        # read all entries in forecast file 
        forecast_file = f'{cfg.raw_data_path}/{cfg.forecast_file}'
        df = pd.read_csv(forecast_file)
        
        # first filter based on product ids and ceil the demand forecast 
        df = df[df['item'].isin(product_ids)]
        df['units'] = df['units'].apply(math.ceil)

        mle_params = {}        
        for id in product_ids:
            df_item = df[(df['item'] == id)]
            observations = np.array(df_item['units'])
            alpha, loc, scale = stats.gamma.fit(observations)
            mle_params[id] = {"alpha": alpha, "loc": loc, "scale": scale}
            # generating data is as simple as 
            # data = stats.gamma.rvs(alpha, loc=loc, scale=scale, size=100)   
        
        return cls(
            config = cfg,
            products = products,
            product_mle_params = mle_params,
        )
    
    def write_to_file(self, fpath: str):
        cfg = self.config._asdict()
        del cfg['raw_data_path']
        del cfg['instance_path']
        del cfg['num_time_points']
        del cfg['location_id']
        del cfg['items_file'] 
        del cfg['forecast_file'] 
        del cfg['open_positions_file']
        data = {
            "config": cfg,
            "products": [p.to_json() for p in self.products],
            "gamma_params": self.product_mle_params
        }
        with open(fpath, "w") as outfile:
            json.dump(data, outfile, indent=4)
        
    
def main():
    """Prepare the data to be written to file"""
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.DEBUG)
    cfg = Config()
    log.info('starting data creation.')
    static_data = StaticData.build(cfg)
    fpath = os.path.join(cfg.instance_path, cfg.instance_name + '.json')
    static_data.write_to_file(fpath)
    log.info(f'wrote instance to {fpath}.')


if __name__ == '__main__':
    main()