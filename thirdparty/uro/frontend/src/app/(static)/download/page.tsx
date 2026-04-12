"use client";

import React, { useEffect, useState } from 'react';
import { InlineLink } from '~/components/link';
import { urls } from '~/environment';
import { Section, SectionTitle } from '~/app/(static)/section';
import { Footer } from '~/app/footer';
import { api } from '~/api';
import { useListDownloads } from '~/hooks/downloads';
import { DownloadsTable, SharedFile } from '~/components/downloads-table';

export default function AboutPage() {
  {
  const sharedFiles = useListDownloads();
  const file_list = sharedFiles?.data?.data?.files ?? [];

  return (
    <main className="mx-auto flex size-full max-w-screen-lg grow flex-col pt-8 lg:pt-16">
      <div className="mx-auto flex w-full max-w-screen-md flex-col gap-4 px-4 text-lg">
        <Section>
          <SectionTitle>Download</SectionTitle>
          <p>
            <b>V-Sekai</b> is currently in open testing. Please check back here
            later for more information.
          </p>
          <DownloadsTable data={file_list} />
        </Section>
        <Section>
          <SectionTitle>Sign up to receive updates</SectionTitle>
          <p>
            At the moment, we have functional prototypes, but are still deep in
            development. We want to ensure that our formal release is
            as polished as it can be.
          </p>
          <p>
            <InlineLink href="/sign-up">Register now</InlineLink> to reserve
            your name and to receive updates on future major releases. You
            can also join{" "}
            <InlineLink href={urls.discord}>V-Sekai Discord Server</InlineLink>
            or follow us on{" "}
            <InlineLink href={urls.twitter}>Twitter</InlineLink> for updates on
            the project.
          </p>
        </Section>
      </div>
      <Footer />
    </main>
  );
};
};
