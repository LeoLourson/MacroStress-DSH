import ssl

import httpx

from mst.data.cli import client


def transport(http, url):
    return http._transport_for_url(httpx.URL(url))


def test_every_host_is_verified():
    with client() as http:
        for url in (
            "https://rosstat.gov.ru/storage/a.xlsx",
            "https://www.rosstat.gov.ru/b.xlsx",
            "https://www.cbr.ru/scripts/XML_dynamic.asp",
            "https://iss.moex.com/iss/index.json",
            "https://api.stlouisfed.org/fred/series/observations",
        ):
            assert transport(http, url)._pool._ssl_context.verify_mode == ssl.CERT_REQUIRED


def test_rosstat_alone_gets_the_mintsifry_roots():
    with client() as http:
        rosstat = transport(http, "https://rosstat.gov.ru/storage/a.xlsx")
        cbr = transport(http, "https://www.cbr.ru/scripts/XML_dynamic.asp")
        assert rosstat is not cbr
        plain = {cert.get("subject") for cert in cbr._pool._ssl_context.get_ca_certs()}
        extra = {
            cert.get("subject") for cert in rosstat._pool._ssl_context.get_ca_certs()
        } - plain
        assert any("Russian Trusted" in str(subject) for subject in extra)


def test_lookalike_hosts_do_not_get_the_extra_roots():
    with client() as http:
        plain = transport(http, "https://www.cbr.ru/x")
        for url in ("https://evilrosstat.gov.ru/a.xlsx", "https://rosstat.gov.ru.example.com/a.xlsx"):
            assert transport(http, url) is plain
