import sys
sys.path.insert(0, '/var/www/html/prod')
sys.path.append('/usr/local/venvs/prod_env/lib/python3.7/site-packages/')
from prod import app as application
