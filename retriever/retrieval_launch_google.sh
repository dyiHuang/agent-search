
api_key="AIzaSyCiuO5seGm0aVy2v_21NR0dZm63tzdz0ic" # put your google custom API key here (https://developers.google.com/custom-search/v1/overview)
cse_id="e347e6869d0c84059" # put your google cse API key here (https://developers.google.com/custom-search/v1/overview)

python ../search_r1/search/google_search_server.py --api_key $api_key \
                                            --topk 5 \
                                            --cse_id $cse_id \
                                            --snippet_only
