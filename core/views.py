from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie

from .services import fetch_eurusd_data, generate_replay_video


@require_http_methods(['GET', 'POST'])
@ensure_csrf_cookie
def replay_home(request):
    if request.method == 'GET':
        return render(request, 'index.html')

    try:
        df = fetch_eurusd_data()
        if df.empty:
            return JsonResponse(
                {'error': 'No EUR/USD market data available. Please try again later.'},
                status=422,
            )

        video_path = generate_replay_video(df)
        video_url = f"{settings.MEDIA_URL}replays/{video_path.name}"

        return JsonResponse({'video_url': video_url})
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=422)
    except RuntimeError as exc:
        return JsonResponse({'error': str(exc)}, status=500)
    except Exception:
        return JsonResponse(
            {'error': 'Video generation failed. Please try again.'},
            status=500,
        )
