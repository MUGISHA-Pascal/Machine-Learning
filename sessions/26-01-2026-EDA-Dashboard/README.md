# Vehicle ML Dashboard

This is a Django-based dashboard for vehicle data analysis featuring frequency tables and Bootstrap-styled UI.

## Project Structure

```
vehicle_ml_dashboard/
├── config/                  # Django project settings & root URL config
├── vehicles/                # Main app
│   ├── dashboard.py         # Data analysis functions (frequency tables)
│   ├── views.py             # Dashboard view
│   ├── urls.py              # App URL routes
│   └── templates/vehicles/  # Bootstrap-styled HTML templates
├── dummy_data/              # Vehicle CSV dataset (1000 records)
├── manage.py
└── requirements.txt
```

## Installation

```bash
cd vehicle_ml_dashboard
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Run

```bash
python manage.py migrate
python manage.py runserver
```

Visit **http://127.0.0.1:8000/**

## Tech Stack

- **Django 6** — Web framework
- **Pandas** — Data manipulation
- **Plotly** — Charting
- **Bootstrap 5** — UI stylinggg
