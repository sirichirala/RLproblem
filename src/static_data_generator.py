# reads raw data, processes it and generates demand scenarios for each product 

import logging 
import json 
import pandas as pd 
import typing 
import os
import math
import json

import random
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
    instance_name: str = 'static_instance'
    num_time_points: int = 50
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
    demand_data: typing.Dict[str, int]
    ramping_factor: typing.Dict[int, int]
    F: typing.Dict[str, int]
    H: typing.Dict[str, int]
    G: typing.Dict[str, int]
    alpha: int

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
        df['units'] = df['units'].apply(math.ceil).abs()

        mle_params = {}
        demand_data = {}        
        for id in product_ids:
            df_item = df[(df['item'] == id)]
            observations = np.array(df_item['units'])
            alpha, loc, scale = stats.gamma.fit(observations)
            mle_params[id] = {"alpha": alpha, "loc": loc, "scale": scale}
            # generating data is as simple as 
            # data = stats.gamma.rvs(alpha, loc=loc, scale=scale, size=100)
            demand = stats.gamma.rvs(alpha, loc=loc, scale=scale, size=cfg.num_time_points)
            demand = np.ceil(demand).astype(int)
            for week in range(cfg.num_time_points):
                key = f"{id}_{week}"  # Create a string representation of the key since key cant be tuple
                demand_data[key] = int(demand[week])
        
        """
        1. Ramping factor for each product.
        ramping_factor = {f"{id}" : random.randint(5, 15) for id in product_ids}
        """
        ramping_factor = {
        "20052773001": 2,
        "20052773002": 1,
        "20052773003": 1,
        "20052773004": 1,
        "20052773005": 1,
        "20052773006": 2,
        "20052773007": 1,
        "20052773008": 1,
        "20064068002": 4,
        "20064068003": 5,
        "20064068004": 2,
        "20064068005": 5,
        "20064068008": 3,
        "20064068009": 4,
        "20064068010": 6,
        "20064068011": 5,
        "20064068013": 4,
        "20064068014": 7,
        "20064068015": 3,
        "20064068016": 4,
        "20064068017": 4
        }

        """
        Cost Construction:
            - F (penalty on failing to meet demand)
                     F = {f"{id}" : random.uniform(30, 40) for id in product_ids}
            - H (penalty on overstocking inventory)
                     H = {f"{id}" : F[f"{id}"]/2 for id in product_ids}
            - G (Reward for maintaining recommended safety stock)
                     G = {f"{id}" : F[f"{id}"]/3 for id in product_ids}
            - alpha (Penalty for violating proportionality)
                     alpha = 25
        """

        F = {f"{id}" : random.uniform(30, 40) for id in product_ids}
        H = {f"{id}" : F[f"{id}"]/2 for id in product_ids}
        G = {f"{id}" : F[f"{id}"]/3 for id in product_ids}

        F = {
        "20052773001": 39.941893645681944,
        "20052773002": 35.62418087896188,
        "20052773003": 39.28131343323098,
        "20052773004": 37.390625382372406,
        "20052773005": 33.96974390290346,
        "20052773006": 30.475698378672593,
        "20052773007": 35.28321763738729,
        "20052773008": 32.200197017293114,
        "20064068002": 31.223413378866,
        "20064068003": 32.156026169786095,
        "20064068004": 39.27856441263787,
        "20064068005": 39.05259758940265,
        "20064068008": 32.05908829632462,
        "20064068009": 31.731439664639105,
        "20064068010": 31.505601858484727,
        "20064068011": 36.1772862489711,
        "20064068013": 36.933300082374224,
        "20064068014": 39.01154198468994,
        "20064068015": 33.45874336264806,
        "20064068016": 39.831010014190056,
        "20064068017": 36.36427265171041
        }
        H = {
        "20052773001": 19.970946822840972,
        "20052773002": 17.81209043948094,
        "20052773003": 19.64065671661549,
        "20052773004": 18.695312691186203,
        "20052773005": 16.98487195145173,
        "20052773006": 15.237849189336297,
        "20052773007": 17.641608818693644,
        "20052773008": 16.100098508646557,
        "20064068002": 15.611706689433,
        "20064068003": 16.078013084893048,
        "20064068004": 19.639282206318935,
        "20064068005": 19.526298794701326,
        "20064068008": 16.02954414816231,
        "20064068009": 15.865719832319552,
        "20064068010": 15.752800929242364,
        "20064068011": 18.08864312448555,
        "20064068013": 18.466650041187112,
        "20064068014": 19.50577099234497,
        "20064068015": 16.72937168132403,
        "20064068016": 19.915505007095028,
        "20064068017": 18.182136325855204
        }
        G = {
        "20052773001": 13.313964548560648,
        "20052773002": 11.87472695965396,
        "20052773003": 13.093771144410326,
        "20052773004": 12.463541794124135,
        "20052773005": 11.323247967634487,
        "20052773006": 10.158566126224198,
        "20052773007": 11.761072545795763,
        "20052773008": 10.73339900576437,
        "20064068002": 10.407804459622,
        "20064068003": 10.718675389928698,
        "20064068004": 13.092854804212623,
        "20064068005": 13.017532529800883,
        "20064068008": 10.686362765441539,
        "20064068009": 10.577146554879702,
        "20064068010": 10.501867286161575,
        "20064068011": 12.0590954163237,
        "20064068013": 12.311100027458075,
        "20064068014": 13.00384732822998,
        "20064068015": 11.152914454216019,
        "20064068016": 13.277003338063352,
        "20064068017": 12.121424217236802
        }
        alpha = 25

            
        return cls(
            config = cfg,
            products = products,
            product_mle_params = mle_params,
            ramping_factor = ramping_factor,
            demand_data=demand_data,
            F = F,
            H = H,
            G = G,
            alpha = 25
        )
    
    def write_to_file(self, fpath: str):
        cfg = self.config._asdict()
        del cfg['raw_data_path']
        del cfg['instance_path']
        del cfg['location_id']
        del cfg['items_file'] 
        del cfg['forecast_file'] 
        del cfg['open_positions_file']
        data = {
            "config": cfg,
            "products": [p.to_json() for p in self.products],
            "gamma_params": self.product_mle_params,
            "ramping_factor": self.ramping_factor,
            "demand_data": self.demand_data,
            "fail_demand_cost": self.F,
            "overstocking_cost": self.H,
            "reward_recommended_stock": self.G,
            "proportionality_cost": self.alpha

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