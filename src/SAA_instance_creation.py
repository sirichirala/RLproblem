#create the SAA ic and static instance for MILP from the generated ic and static instances.
import logging
import json
import typing
import os

import numpy as np

log = logging.getLogger(__name__)


class Config(typing.NamedTuple):
    """Configuration of SAA instance to be generated for MILP
    """
    instance_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'SAA'))
    
class IcInstance(typing.NamedTuple):
    config: Config

    @classmethod 
    def build(cls, cfg: Config):
        total_instances = 0
        on_hand_inventory={}
        for filename in os.listdir(cfg.instance_path):
            if filename.startswith("ic_"):
                file_path = os.path.join(cfg.instance_path, filename)
                total_instances +=1
                with open(file_path, "r") as file:
                    data = json.load(file)
                    products = data["products"]
                    for product in products:
                        product_id = product["id"]
                        if product_id not in on_hand_inventory:
                            on_hand_inventory[product_id] = 0
                        on_hand_inventory[product_id] += product["on_hand_inventory"]
        on_hand_inventory = {key: np.floor(value / total_instances) for key, value in on_hand_inventory.items()}

        products = [{"id": key, "on_hand_inventory": value} for key, value in on_hand_inventory.items()]
        data = {
            "config": {
                "instance_name": "ic_instance"
            },
        "products": products
        }
        fpath = os.path.join(cfg.instance_path, "ic_instance.json")
        with open(fpath, "w") as json_file:
            json.dump(data, json_file, indent=4)


class StaticInstance(typing.NamedTuple):
    config: Config

    @classmethod 
    def build(cls, cfg: Config):
        total_instances = 0
        demand_data={}
        for filename in os.listdir(cfg.instance_path):
            if filename.startswith("static_"):
                file_path = os.path.join(cfg.instance_path, filename)
                total_instances +=1
                with open(file_path, "r") as file:
                    data = json.load(file)
                    demands=data["demand_data"]
                    for key, value in demands.items():
                        if key not in demand_data:
                            demand_data[key] = 0
                        demand_data[key] += value
        demand_data = {key: np.floor(value / total_instances) for key, value in demand_data.items()}
        
        fpath = os.path.join(cfg.instance_path, "static_instance_0.json")
        with open(fpath, "r") as file:
            data = json.load(file)
            data["demand_data"] = demand_data
        fpath = os.path.join(cfg.instance_path, "static_instance.json")
        with open(fpath, "w") as file:
            json.dump(data, file, indent=4)
    
def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.DEBUG)
    
    cfg = Config()
    log.info('Building ic data instance for SAA')
    ic_data = IcInstance.build(cfg)
    log.info('ic instance done')
    log.info('Building static data instance for SAA')
    static_data = StaticInstance.build(cfg)
    log.info('static instance done')



if __name__ == '__main__':
    main()
